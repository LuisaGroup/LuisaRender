//
// Created by Mike Smith on 2022/9/14.
//

#include <util/sampling.h>
#include <base/geometry.h>
#include <base/pipeline.h>

namespace luisa::render {

void Geometry::build(CommandBuffer &command_buffer, luisa::span<const Shape *const> shapes,
                     float init_time, AccelUsageHint hint) noexcept {
    _accel = _pipeline.device().create_accel(hint);
    for (auto shape : shapes) { _process_shape(command_buffer, shape, init_time); }
    _instance_buffer = _pipeline.device().create_buffer<Shape::Handle>(_instances.size());
    command_buffer << _instance_buffer.copy_from(_instances.data())
                   << _accel.build();
}

void Geometry::_process_shape(CommandBuffer &command_buffer, const Shape *shape, float init_time,
                              luisa::optional<bool> overridden_two_sided,
                              const Surface *overridden_surface, const Light *overridden_light) noexcept {

    auto surface = overridden_surface == nullptr ? shape->surface() : overridden_surface;
    auto light = overridden_light == nullptr ? shape->light() : overridden_light;

    if (shape->is_mesh()) {
        if (shape->deformable()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Deformable meshes are not yet supported.");
        }
        auto iter = _meshes.find(shape);
        if (iter == _meshes.end()) {
            auto mesh_geom = [&] {
                auto vertices = shape->vertices();
                auto triangles = shape->triangles();
                if (vertices.empty() || triangles.empty()) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION("Found mesh without vertices.");
                }
                auto hash = luisa::detail::murmur2_hash64(vertices.data(), vertices.size_bytes(), Hash64::default_seed);
                hash = luisa::detail::murmur2_hash64(triangles.data(), triangles.size_bytes(), hash);
                auto [cache_iter, non_existent] = _mesh_cache.try_emplace(hash, MeshGeometry{});
                if (!non_existent) { return cache_iter->second; }

                // create mesh
                auto vertex_buffer = _pipeline.create<Buffer<Shape::Vertex>>(vertices.size());
                auto triangle_buffer = _pipeline.create<Buffer<Triangle>>(triangles.size());
                auto mesh = _pipeline.create<Mesh>(*vertex_buffer, *triangle_buffer, shape->build_hint());
                command_buffer << vertex_buffer->copy_from(vertices.data())
                               << triangle_buffer->copy_from(triangles.data())
                               << mesh->build()
                               << compute::commit();
                auto vertex_buffer_id = _pipeline.register_bindless(vertex_buffer->view());
                auto triangle_buffer_id = _pipeline.register_bindless(triangle_buffer->view());
                // compute alias table
                luisa::vector<float> triangle_areas(triangles.size());
                std::transform(triangles.cbegin(), triangles.cend(), triangle_areas.begin(), [vertices](auto t) noexcept {
                    auto p0 = vertices[t.i0].pos;
                    auto p1 = vertices[t.i1].pos;
                    auto p2 = vertices[t.i2].pos;
                    return std::abs(length(cross(p1 - p0, p2 - p0)));
                });
                auto [alias_table, pdf] = create_alias_table(triangle_areas);
                auto [alias_table_buffer_view, alias_buffer_id] = _pipeline.arena_buffer<AliasEntry>(alias_table.size());
                auto [pdf_buffer_view, pdf_buffer_id] = _pipeline.arena_buffer<float>(pdf.size());
                LUISA_ASSERT(triangle_buffer_id - vertex_buffer_id == Shape::Handle::triangle_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(alias_buffer_id - vertex_buffer_id == Shape::Handle::alias_table_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(pdf_buffer_id - vertex_buffer_id == Shape::Handle::pdf_buffer_id_offset, "Invalid.");
                command_buffer << alias_table_buffer_view.copy_from(alias_table.data())
                               << pdf_buffer_view.copy_from(pdf.data())
                               << compute::commit();
                return cache_iter->second = {mesh, vertex_buffer_id};
            }();
            // assign mesh data
            MeshData mesh{};
            mesh.resource = mesh_geom.resource;
            mesh.geometry_buffer_id_base = mesh_geom.buffer_id_base;
            mesh.two_sided = shape->two_sided().value_or(false);
            mesh.shadow_term = shape->shadow_terminator_factor();
            iter = _meshes.emplace(shape, mesh).first;
        }
        auto mesh = iter->second;
        auto two_sided = mesh.two_sided;
        two_sided = overridden_two_sided.value_or(two_sided);
        auto instance_id = static_cast<uint>(_accel.size());
        auto [t_node, is_static] = _transform_tree.leaf(shape->transform());
        InstancedTransform inst_xform{t_node, instance_id};
        if (!is_static) { _dynamic_transforms.emplace_back(inst_xform); }
        auto object_to_world = inst_xform.matrix(init_time);
        _accel.emplace_back(*mesh.resource, object_to_world, true);

        // create instance
        auto properties = 0u;
        auto surface_tag = 0u;
        auto light_tag = 0u;
        if (two_sided) { properties |= Shape::property_flag_two_sided; }
        if (surface != nullptr && !surface->is_null()) {
            surface_tag = _pipeline.register_surface(command_buffer, surface);
            properties |= Shape::property_flag_has_surface;
        }
        if (light != nullptr && !light->is_null()) {
            light_tag = _pipeline.register_light(command_buffer, light);
            properties |= Shape::property_flag_has_light;
        }
        _instances.emplace_back(Shape::Handle::encode(
            mesh.geometry_buffer_id_base,
            properties, surface_tag, light_tag,
            mesh.resource->triangle_count(),
            mesh.shadow_term));
        if (properties & Shape::property_flag_has_light) {
            _instanced_lights.emplace_back(Light::Handle{
                .instance_id = instance_id,
                .light_tag = light_tag});
        }
    } else {
        _transform_tree.push(shape->transform());
        for (auto child : shape->children()) {
            _process_shape(command_buffer, child, init_time,
                           shape->two_sided(), surface, light);
        }
        _transform_tree.pop(shape->transform());
    }
}

bool Geometry::update(CommandBuffer &command_buffer, float time) noexcept {
    auto updated = false;
    if (!_dynamic_transforms.empty()) {
        updated = true;
        if (_dynamic_transforms.size() < 128u) {
            for (auto t : _dynamic_transforms) {
                _accel.set_transform_on_update(
                    t.instance_id(), t.matrix(time));
            }
        } else {
            ThreadPool::global().parallel(
                _dynamic_transforms.size(),
                [this, time](auto i) noexcept {
                    auto t = _dynamic_transforms[i];
                    _accel.set_transform_on_update(
                        t.instance_id(), t.matrix(time));
                });
            ThreadPool::global().synchronize();
        }
        command_buffer << _accel.build();
    }
    return updated;
}

Var<Hit> Geometry::trace_closest(const Var<Ray> &ray) const noexcept {
    return _accel.trace_closest(ray);
}

Var<bool> Geometry::trace_any(const Var<Ray> &ray) const noexcept {
    return _accel.trace_any(ray);
}

luisa::unique_ptr<Interaction> Geometry::interaction(const Var<Ray> &ray, const Var<Hit> &hit) const noexcept {
    using namespace luisa::compute;
    Interaction it;
    $if(hit->miss()) {
        it = Interaction{-ray->direction()};
    }
    $else {
        auto shape = instance(hit.inst);
        auto m = instance_to_world(hit.inst);
        auto n = transpose(inverse(make_float3x3(m)));
        auto tri = triangle(shape, hit.prim);
        auto uvw = make_float3(1.0f - hit.bary.x - hit.bary.y, hit.bary);
        auto attrib = shading_point(shape, tri, uvw, m, n);
        auto wo = -ray->direction();
        it = Interaction{std::move(shape), hit.inst, hit.prim, wo, attrib};
    };
    return luisa::make_unique<Interaction>(std::move(it));
}

Var<Shape::Handle> Geometry::instance(Expr<uint> index) const noexcept {
    return _instance_buffer.read(index);
}

Float4x4 Geometry::instance_to_world(Expr<uint> index) const noexcept {
    return _accel.instance_transform(index);
}

Var<Triangle> Geometry::triangle(const Var<Shape::Handle> &instance, Expr<uint> index) const noexcept {
    return _pipeline.buffer<Triangle>(instance->triangle_buffer_id()).read(index);
}

ShadingAttribute Geometry::shading_point(const Var<Shape::Handle> &instance, const Var<Triangle> &triangle,
                                         const Var<float3> &bary, const Var<float4x4> &shape_to_world,
                                         const Var<float3x3> &shape_to_world_normal) const noexcept {
    auto v_buffer = instance->vertex_buffer_id();
    auto v0 = _pipeline.buffer<Shape::Vertex>(v_buffer).read(triangle.i0);
    auto v1 = _pipeline.buffer<Shape::Vertex>(v_buffer).read(triangle.i1);
    auto v2 = _pipeline.buffer<Shape::Vertex>(v_buffer).read(triangle.i2);
    auto p0 = make_float3(shape_to_world * make_float4(v0->position(), 1.f));
    auto p1 = make_float3(shape_to_world * make_float4(v1->position(), 1.f));
    auto p2 = make_float3(shape_to_world * make_float4(v2->position(), 1.f));
    auto p = bary.x * p0 + bary.y * p1 + bary.z * p2;
    auto c = cross(p1 - p0, p2 - p0);
    auto area = 0.5f * length(c);
    auto ng = normalize(c);
    auto uv = bary.x * v0->uv() + bary.y * v1->uv() + bary.z * v2->uv();
    auto n0 = normalize(shape_to_world_normal * v0->normal());
    auto n1 = normalize(shape_to_world_normal * v1->normal());
    auto n2 = normalize(shape_to_world_normal * v2->normal());
    auto ns = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
    auto tangent_local = bary.x * v0->tangent() + bary.y * v1->tangent() + bary.z * v2->tangent();
    auto tangent = normalize(shape_to_world_normal * tangent_local);
    // offset p to fake surface for the shadow terminator
    // reference: Ray Tracing Gems 2, Chap. 4
    auto temp_u = p - p0;
    auto temp_v = p - p1;
    auto temp_w = p - p2;
    auto dot_u = min(dot(temp_u, n0), 0.f);
    auto dot_v = min(dot(temp_v, n1), 0.f);
    auto dot_w = min(dot(temp_w, n2), 0.f);
    auto shadow_term = instance->shadow_terminator_factor();
    auto dp = bary.x * (temp_u - dot_u * n0) + bary.y * (temp_v - dot_v * n1) + bary.z * (temp_w - dot_w * n2);
    return {.pg = p, .ng = ng, .ps = p + shadow_term * dp, .ns = ns, .tangent = tangent, .uv = uv, .area = area};
}

}// namespace luisa::render

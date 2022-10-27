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
                              const Surface *overridden_surface, const Light *overridden_light,
                              bool overriden_visible) noexcept {

    auto surface = overridden_surface == nullptr ? shape->surface() : overridden_surface;
    auto light = overridden_light == nullptr ? shape->light() : overridden_light;
    auto visible = overriden_visible && shape->visible();

    if (shape->is_mesh()) {
        if (shape->deformable()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Deformable meshes are not yet supported.");
        }
        auto mesh = [&] {
            if (auto iter = _meshes.find(shape); iter != _meshes.end()) {
                return iter->second;
            }
            auto mesh_geom = [&] {
                auto vertices = shape->vertices();
                auto triangles = shape->triangles();
                if (vertices.empty() || triangles.empty()) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION("Found mesh without vertices.");
                }
                auto hash = luisa::detail::murmur2_hash64(vertices.data(), vertices.size_bytes(), Hash64::default_seed);
                hash = luisa::detail::murmur2_hash64(triangles.data(), triangles.size_bytes(), hash);
                if (auto mesh_iter = _mesh_cache.find(hash);
                    mesh_iter != _mesh_cache.end()) {
                    return mesh_iter->second;
                }
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
                    auto p0 = make_float3(vertices[t.i0].compressed_p[0], vertices[t.i0].compressed_p[1], vertices[t.i0].compressed_p[2]);
                    auto p1 = make_float3(vertices[t.i1].compressed_p[0], vertices[t.i1].compressed_p[1], vertices[t.i1].compressed_p[2]);
                    auto p2 = make_float3(vertices[t.i2].compressed_p[0], vertices[t.i2].compressed_p[1], vertices[t.i2].compressed_p[2]);
                    return std::abs(length(cross(p1 - p0, p2 - p0)));
                });
                auto [alias_table, pdf] = create_alias_table(triangle_areas);
                auto [alias_table_buffer_view, alias_buffer_id] = _pipeline.bindless_arena_buffer<AliasEntry>(alias_table.size());
                auto [pdf_buffer_view, pdf_buffer_id] = _pipeline.bindless_arena_buffer<float>(pdf.size());
                LUISA_ASSERT(triangle_buffer_id - vertex_buffer_id == Shape::Handle::triangle_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(alias_buffer_id - vertex_buffer_id == Shape::Handle::alias_table_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(pdf_buffer_id - vertex_buffer_id == Shape::Handle::pdf_buffer_id_offset, "Invalid.");
                command_buffer << alias_table_buffer_view.copy_from(alias_table.data())
                               << pdf_buffer_view.copy_from(pdf.data())
                               << compute::commit();
                auto geom = MeshGeometry{mesh, vertex_buffer_id};
                _mesh_cache.emplace(hash, geom);
                return geom;
            }();
            // assign mesh data
            MeshData mesh_data{};
            mesh_data.resource = mesh_geom.resource;
            mesh_data.shadow_term = shape->shadow_terminator_factor();
            mesh_data.geometry_buffer_id_base = mesh_geom.buffer_id_base;
            mesh_data.has_normal = shape->has_normal();
            mesh_data.has_uv = shape->has_uv();
            _meshes.emplace(shape, mesh_data);
            return mesh_data;
        }();
        auto instance_id = static_cast<uint>(_accel.size());
        auto [t_node, is_static] = _transform_tree.leaf(shape->transform());
        InstancedTransform inst_xform{t_node, instance_id};
        if (!is_static) { _dynamic_transforms.emplace_back(inst_xform); }
        auto object_to_world = inst_xform.matrix(init_time);
        _accel.emplace_back(*mesh.resource, object_to_world, visible);

        // create instance
        auto properties = 0u;
        auto surface_tag = 0u;
        auto light_tag = 0u;
        if (surface != nullptr && !surface->is_null()) {
            surface_tag = _pipeline.register_surface(command_buffer, surface);
            properties |= Shape::property_flag_has_surface;
        }
        if (light != nullptr && !light->is_null()) {
            light_tag = _pipeline.register_light(command_buffer, light);
            properties |= Shape::property_flag_has_light;
        }
        if (mesh.has_normal) { properties |= Shape::property_flag_has_normal; }
        if (mesh.has_uv) { properties |= Shape::property_flag_has_uv; }
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
                           surface, light, visible);
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
    $if(!hit->miss()) {
        auto shape = instance(hit.inst);
        auto m = instance_to_world(hit.inst);
        auto n = transpose(inverse(make_float3x3(m)));
        auto tri = triangle(shape, hit.prim);
        auto uvw = make_float3(1.0f - hit.bary.x - hit.bary.y, hit.bary);
        auto attrib = shading_point(shape, tri, uvw, m, n);
        it = {std::move(shape), hit.inst, hit.prim, attrib, dot(ray->direction(), attrib.ng) > 0.0f};
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

[[nodiscard]] static auto _compute_tangent(
    Expr<float3> p0, Expr<float3> p1, Expr<float3> p2,
    Expr<float2> uv0, Expr<float2> uv1, Expr<float2> uv2) noexcept {
    static Callable impl = [](Float3 p0, Float3 p1, Float3 p2,
                              Float2 uv0, Float2 uv1, Float2 uv2) noexcept {
        auto difference_of_products = [](auto a, auto b, auto c, auto d) noexcept {
            auto cd = c * d;
            auto differenceOfProducts = a * b - cd;
            auto error = -c * d + cd;
            return differenceOfProducts + error;
        };
        auto duv02 = uv0 - uv2;
        auto duv12 = uv1 - uv2;
        auto dp02 = p0 - p2;
        auto dp12 = p1 - p2;
        auto det = difference_of_products(duv02.x, duv12.y, duv02.y, duv12.x);
        auto dpdu = def(make_float3());
        auto dpdv = def(make_float3());
        auto degenerate_uv = abs(det) < 1e-8f;
        $if(!degenerate_uv) {
            // Compute triangle $\dpdu$ and $\dpdv$ via matrix inversion
            auto invdet = 1.f / det;
            dpdu = difference_of_products(duv12.y, dp02, duv02.y, dp12) * invdet;
            dpdv = difference_of_products(duv02.x, dp12, duv12.x, dp02) * invdet;
        };
        // Handle degenerate triangle $(u,v)$ parameterization or partial derivatives
        $if(degenerate_uv | length_squared(cross(dpdu, dpdv)) == 0.f) {
            auto n = cross(p2 - p0, p1 - p0);
            auto b = ite(abs(n.x) > abs(n.z),
                         make_float3(-n.y, n.x, 0.0f),
                         make_float3(0.0f, -n.z, n.y));
            dpdu = cross(b, n);
        };
        return dpdu;
    };
    return impl(p0, p1, p2, uv0, uv1, uv2);
}

ShadingAttribute Geometry::shading_point(const Var<Shape::Handle> &instance, const Var<Triangle> &triangle,
                                         const Var<float3> &bary, const Var<float4x4> &shape_to_world,
                                         const Var<float3x3> &shape_to_world_normal) const noexcept {
    auto v_buffer = instance->vertex_buffer_id();
    // The following calculations are all in the object space.
    auto v0 = _pipeline.buffer<Shape::Vertex>(v_buffer).read(triangle.i0);
    auto v1 = _pipeline.buffer<Shape::Vertex>(v_buffer).read(triangle.i1);
    auto v2 = _pipeline.buffer<Shape::Vertex>(v_buffer).read(triangle.i2);
    auto p0 = v0->position();
    auto p1 = v1->position();
    auto p2 = v2->position();
    auto p = bary.x * p0 + bary.y * p1 + bary.z * p2;
    auto ng = cross(p1 - p0, p2 - p0);
    auto uv0 = ite(instance->has_uv(), v0->uv(), make_float2());
    auto uv1 = ite(instance->has_uv(), v1->uv(), make_float2());
    auto uv2 = ite(instance->has_uv(), v2->uv(), make_float2());
    auto uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
    auto tangent = _compute_tangent(p0, p1, p2, uv0, uv1, uv2);
    auto n0 = ite(instance->has_normal(), v0->normal(), ng);
    auto n1 = ite(instance->has_normal(), v1->normal(), ng);
    auto n2 = ite(instance->has_normal(), v2->normal(), ng);
    auto ns = bary.x * n0 + bary.y * n1 + bary.z * n2;
    // offset p to fake surface for the shadow terminator
    // reference: Ray Tracing Gems 2, Chap. 4
    auto temp_u = p - p0;
    auto temp_v = p - p1;
    auto temp_w = p - p2;
    auto shadow_term = instance->shadow_terminator_factor();
    auto dp = bary.x * (temp_u - min(dot(temp_u, n0), 0.f) * n0) +
              bary.y * (temp_v - min(dot(temp_v, n1), 0.f) * n1) +
              bary.z * (temp_w - min(dot(temp_w, n2), 0.f) * n2);
    auto ps = p + shadow_term * dp;
    // Now, let's go into the world space.
    // A * (x, 1.f) - A * (y, 1.f) = A * (x - y, 0.f)
    auto c = cross(make_float3(shape_to_world * make_float4(p1 - p0, 0.f)),
                   make_float3(shape_to_world * make_float4(p2 - p0, 0.f)));
    auto area = 0.5f * length(c);
    return {.pg = make_float3(shape_to_world * make_float4(p, 1.f)),
            .ng = normalize(shape_to_world_normal * ng),
            .ps = make_float3(shape_to_world * make_float4(ps, 1.f)),
            .ns = normalize(shape_to_world_normal * ns),
            .tangent = normalize(shape_to_world_normal * tangent),
            .uv = uv,
            .area = area};
}

}// namespace luisa::render

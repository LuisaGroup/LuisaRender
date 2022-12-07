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
    _instance_buffer = _pipeline.device().create_buffer<uint4>(_instances.size());
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
                auto [vertices, triangles] = shape->mesh();
                LUISA_ASSERT(!vertices.empty() && !triangles.empty(), "Empty mesh.");
                auto hash = luisa::detail::murmur2_hash64(vertices.data(), vertices.size_bytes(), Hash64::default_seed);
                hash = luisa::detail::murmur2_hash64(triangles.data(), triangles.size_bytes(), hash);
                if (auto mesh_iter = _mesh_cache.find(hash);
                    mesh_iter != _mesh_cache.end()) {
                    return mesh_iter->second;
                }
                // create mesh
                auto vertex_buffer = _pipeline.create<Buffer<Vertex>>(vertices.size());
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
                for (auto i = 0u; i < triangles.size(); i++) {
                    auto t = triangles[i];
                    auto p0 = vertices[t.i0].position();
                    auto p1 = vertices[t.i1].position();
                    auto p2 = vertices[t.i2].position();
                    triangle_areas[i] = std::abs(length(cross(p1 - p0, p2 - p0)));
                }
                auto [alias_table, pdf] = create_alias_table(triangle_areas);
                auto [alias_table_buffer_view, alias_buffer_id] = _pipeline.bindless_arena_buffer<AliasEntry>(alias_table.size());
                auto [pdf_buffer_view, pdf_buffer_id] = _pipeline.bindless_arena_buffer<float>(pdf.size());
                LUISA_ASSERT(triangle_buffer_id - vertex_buffer_id == Shape::Handle::triangle_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(alias_buffer_id - vertex_buffer_id == Shape::Handle::alias_table_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(pdf_buffer_id - vertex_buffer_id == Shape::Handle::pdf_buffer_id_offset, "Invalid.");
                command_buffer << alias_table_buffer_view.copy_from(alias_table.data())
                               << pdf_buffer_view.copy_from(pdf.data());
                command_buffer << compute::commit();
                auto geom = MeshGeometry{mesh, vertex_buffer_id};
                _mesh_cache.emplace(hash, geom);
                return geom;
            }();
            auto encode_fixed_point = [](float x) noexcept {
                return static_cast<uint16_t>(std::clamp(
                    std::round(x * 65535.f), 0.f, 65535.f));
            };
            // assign mesh data
            MeshData mesh_data{
                .resource = mesh_geom.resource,
                .shadow_term = encode_fixed_point(shape->has_vertex_normal() ? shape->shadow_terminator_factor() : 0.f),
                .intersection_offset = encode_fixed_point(shape->intersection_offset_factor()),
                .geometry_buffer_id_base = mesh_geom.buffer_id_base,
                .vertex_properties = shape->vertex_properties()};
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
        auto surface_tag = 0u;
        auto light_tag = 0u;
        auto properties = mesh.vertex_properties;
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
            static_cast<float>(mesh.shadow_term) / 65535.f,
            static_cast<float>(mesh.intersection_offset) / 65535.f));
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

luisa::shared_ptr<Interaction> Geometry::interaction(Expr<uint> inst_id, Expr<uint> prim_id,
                                                     Expr<float3> bary, Expr<float3> wo) const noexcept {
    auto shape = instance(inst_id);
    auto m = instance_to_world(inst_id);
    auto n = transpose(inverse(make_float3x3(m)));
    auto tri = triangle(*shape, prim_id);
    auto attrib = shading_point(*shape, tri, bary, m, n);
    return luisa::make_shared<Interaction>(
        std::move(shape), inst_id, prim_id,
        attrib, dot(wo, attrib.g.n) < 0.0f);
}

luisa::shared_ptr<Interaction> Geometry::interaction(const Var<Ray> &ray, const Var<Hit> &hit) const noexcept {
    using namespace luisa::compute;
    Interaction it;
    $if(!hit->miss()) {
        it = *interaction(hit.inst, hit.prim,
                          make_float3(1.f - hit.bary.x - hit.bary.y, hit.bary),
                          -ray->direction());
    };
    return luisa::make_shared<Interaction>(std::move(it));
}

luisa::shared_ptr<Shape::Handle> Geometry::instance(Expr<uint> index) const noexcept {
    return Shape::Handle::decode(_instance_buffer.read(index));
}

Float4x4 Geometry::instance_to_world(Expr<uint> index) const noexcept {
    return _accel.instance_transform(index);
}

Var<Triangle> Geometry::triangle(const Shape::Handle &instance, Expr<uint> index) const noexcept {
    return _pipeline.buffer<Triangle>(instance.triangle_buffer_id()).read(index);
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

template<typename T>
[[nodiscard]] inline auto interpolate(Expr<float3> uvw,
                                      const T &v0,
                                      const T &v1,
                                      const T &v2) noexcept {
    return uvw.x * v0 + uvw.y * v1 + uvw.z * v2;
}

GeometryAttribute Geometry::geometry_point(const Shape::Handle &instance, const Var<Triangle> &triangle,
                                           const Var<float3> &bary, const Var<float4x4> &shape_to_world,
                                           const Var<float3x3> &shape_to_world_normal) const noexcept {
    auto v_buffer = instance.vertex_buffer_id();
    auto v0 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
    auto v1 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
    auto v2 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
    auto p0 = make_float3(shape_to_world * make_float4(v0->position(), 1.f));
    auto p1 = make_float3(shape_to_world * make_float4(v1->position(), 1.f));
    auto p2 = make_float3(shape_to_world * make_float4(v2->position(), 1.f));
    auto p = interpolate(bary, p0, p1, p2);
    auto c = cross(p1 - p0, p2 - p0);
    auto area = .5f * length(c);
    auto ng = normalize(c);
    return {.p = p, .n = ng, .area = area};
}

ShadingAttribute Geometry::shading_point(const Shape::Handle &instance, const Var<Triangle> &triangle,
                                         const Var<float3> &bary, const Var<float4x4> &shape_to_world,
                                         const Var<float3x3> &shape_to_world_normal) const noexcept {
    auto v_buffer = instance.vertex_buffer_id();
    auto v0 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
    auto v1 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
    auto v2 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
    auto p0 = make_float3(shape_to_world * make_float4(v0->position(), 1.f));
    auto p1 = make_float3(shape_to_world * make_float4(v1->position(), 1.f));
    auto p2 = make_float3(shape_to_world * make_float4(v2->position(), 1.f));
    auto uv0 = ite(instance.has_vertex_uv(), v0->uv(), make_float2(0.f));
    auto uv1 = ite(instance.has_vertex_uv(), v1->uv(), make_float2(0.f));
    auto uv2 = ite(instance.has_vertex_uv(), v2->uv(), make_float2(0.f));
    auto s = _compute_tangent(p0, p1, p2, uv0, uv1, uv2);
    auto p = interpolate(bary, p0, p1, p2);
    auto uv = interpolate(bary, uv0, uv1, uv2);
    auto c = cross(p1 - p0, p2 - p0);
    auto area = .5f * length(c);
    auto ng = normalize(c);
    auto ns = ng;
    auto ps = p;
    auto shadow_term = instance.shadow_terminator_factor();
    $if(instance.has_vertex_normal() & shadow_term > 0.f) {
        auto n0 = normalize(shape_to_world_normal * v0->normal());
        auto n1 = normalize(shape_to_world_normal * v1->normal());
        auto n2 = normalize(shape_to_world_normal * v2->normal());
        ns = normalize(interpolate(bary, n0, n1, n2));
        // offset p to fake surface for the shadow terminator
        // reference: Ray Tracing Gems 2, Chap. 4
        auto temp_u = p - p0;
        auto temp_v = p - p1;
        auto temp_w = p - p2;
        auto dp = interpolate(bary,
                              temp_u - min(dot(temp_u, n0), 0.f) * n0,
                              temp_v - min(dot(temp_v, n1), 0.f) * n1,
                              temp_w - min(dot(temp_w, n2), 0.f) * n2);
        ps = p + shadow_term * dp;
    };
    return {.g = {.p = p,
                  .n = ng,
                  .area = area},
            .ps = ps,
            .ns = ns,
            .tangent = s,
            .uv = uv};
}

}// namespace luisa::render

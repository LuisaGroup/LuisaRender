//
// Created by Mike Smith on 2022/9/14.
//

#include <util/sampling.h>
#include <util/thread_pool.h>
#include <base/geometry.h>
#include <base/pipeline.h>

namespace luisa::render {

void Geometry::build(CommandBuffer &command_buffer,
                     luisa::span<const Shape *const> shapes,
                     float init_time) noexcept {
    // TODO: AccelOption
    _accel = _pipeline.device().create_accel({});
    for (auto i = 0u; i < 3u; ++i) {
        _world_max[i] = -std::numeric_limits<float>::max();
        _world_min[i] = std::numeric_limits<float>::max();
    }
    _triangle_count = 0u;
    for (auto shape : shapes) { _process_shape(command_buffer, shape, init_time, nullptr); }
    LUISA_INFO_WITH_LOCATION("Geometry built with {} triangles.", _triangle_count);
    _instance_buffer = _pipeline.device().create_buffer<uint4>(_instances.size());
    command_buffer << _instance_buffer.copy_from(_instances.data())
                   << _accel.build();
}

void Geometry::_process_shape(
    CommandBuffer &command_buffer, const Shape *shape, float init_time,
    const Surface *overridden_surface,
    const Light *overridden_light,
    const Medium *overridden_medium,
    bool overridden_visible) noexcept {

    auto surface = overridden_surface == nullptr ? shape->surface() : overridden_surface;
    auto light = overridden_light == nullptr ? shape->light() : overridden_light;
    auto medium = overridden_medium == nullptr ? shape->medium() : overridden_medium;
    auto visible = overridden_visible && shape->visible();

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
                auto hash = luisa::hash64(vertices.data(), vertices.size_bytes(), luisa::hash64_default_seed);
                hash = luisa::hash64(triangles.data(), triangles.size_bytes(), hash);
                if (auto mesh_iter = _mesh_cache.find(hash);
                    mesh_iter != _mesh_cache.end()) {
                    return mesh_iter->second;
                }
                // create mesh
                auto vertex_buffer = _pipeline.create<Buffer<Vertex>>(vertices.size());
                auto triangle_buffer = _pipeline.create<Buffer<Triangle>>(triangles.size());
                auto mesh = _pipeline.create<Mesh>(*vertex_buffer, *triangle_buffer, shape->build_option());
                command_buffer << vertex_buffer->copy_from(vertices.data())
                               << triangle_buffer->copy_from(triangles.data())
                               << compute::commit()
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
        auto vertices = shape->mesh().vertices;
        for (auto &v : vertices) {
            _world_max = max(_world_max, make_float3(object_to_world * make_float4(v.position(), 1.f)));
            _world_min = min(_world_min, make_float3(object_to_world * make_float4(v.position(), 1.f)));
        }

        // create instance
        auto surface_tag = 0u;
        auto properties = mesh.vertex_properties;
        if (surface != nullptr && !surface->is_null()) {
            surface_tag = _pipeline.register_surface(command_buffer, surface);
            properties |= Shape::property_flag_has_surface;
            if (_pipeline.surfaces().impl(surface_tag)->maybe_non_opaque()) {
                properties |= Shape::property_flag_maybe_non_opaque;
                _any_non_opaque = true;
            }
        }

        // emplace instance here since we need to know the opaque property
        _accel.emplace_back(*mesh.resource, object_to_world, visible, false);
                            //(properties & Shape::property_flag_maybe_non_opaque) == 0u);

        auto light_tag = 0u;
        auto medium_tag = 0u;
        if (light != nullptr && !light->is_null()) {
            light_tag = _pipeline.register_light(command_buffer, light);
            properties |= Shape::property_flag_has_light;
        }
        if (medium != nullptr && !medium->is_null()) {
            medium_tag = _pipeline.register_medium(command_buffer, medium);
            properties |= Shape::property_flag_has_medium;
        }
        _instances.emplace_back(Shape::Handle::encode(
            mesh.geometry_buffer_id_base,
            properties, surface_tag, light_tag, medium_tag,
            mesh.resource->triangle_count(),
            static_cast<float>(mesh.shadow_term) / 65535.f,
            static_cast<float>(mesh.intersection_offset) / 65535.f));
        if (properties & Shape::property_flag_has_light) {
            _instanced_lights.emplace_back(Light::Handle{
                .instance_id = instance_id,
                .light_tag = light_tag});
        }
        _triangle_count += mesh.resource->triangle_count();
    } else {
        _transform_tree.push(shape->transform());
        for (auto child : shape->children()) {
            _process_shape(command_buffer, child, init_time,
                           surface, light, medium, visible);
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
            global_thread_pool().parallel(
                _dynamic_transforms.size(),
                [this, time](auto i) noexcept {
                    auto t = _dynamic_transforms[i];
                    _accel.set_transform_on_update(
                        t.instance_id(), t.matrix(time));
                });
            global_thread_pool().synchronize();
        }
        command_buffer << _accel.build();
    }
    return updated;
}

Var<Hit> Geometry::trace_closest(const Var<Ray> &ray) const noexcept {
    if (!_any_non_opaque) {
        // happy path
        auto hit = _accel->intersect(ray, {});
        return Var<Hit>{hit.inst, hit.prim, hit.bary};
    }
    auto rq_hit =
        _accel->traverse(ray, {})
            .on_surface_candidate([&](compute::SurfaceCandidate &c) noexcept {
                auto hit = c.hit();
                auto bary = make_float3(1.f - hit.bary.x - hit.bary.y, hit.bary);
                auto it = interaction(hit.inst, hit.prim, bary, -ray->direction());
                $if(it->shape().maybe_non_opaque() & it->shape().has_surface()) {
                    PolymorphicCall<Surface::Closure> call;
                    _pipeline.surfaces().dispatch(it->shape().surface_tag(), [&](auto surface) noexcept {
                        $if(surface->maybe_non_opaque()) {
                            // TODO: pass the correct time
                            surface->closure(call, *it, _pipeline.spectrum()->sample(.5f), -ray->direction(), 1.f, 0.f);
                        };
                    });
                    auto u1 = xxhash32(as<uint3>(ray->origin()));
                    auto u2 = xxhash32(as<uint3>(ray->direction()));
                    auto u = xxhash32(make_uint2(u1, u2)) * 0x1p-32f;
                    call.execute([&](const Surface::Closure *closure) noexcept {
                        // apply opacity map
                        auto alpha_skip = def(false);
                        if (auto o = closure->opacity()) {
                            auto opacity = saturate(*o);
                            alpha_skip = u >= opacity;
                        }
                        $if(!alpha_skip) {
                            c.commit();
                        };
                    });
                }
                $else {
                    c.commit();
                };
            })
            .trace();
    return Var<Hit>{rq_hit.inst, rq_hit.prim, rq_hit.bary};
}

Var<bool> Geometry::trace_any(const Var<Ray> &ray) const noexcept {
    if (!_any_non_opaque) {
        // happy path
        return _accel->intersect_any(ray, {});
    }
    auto rq_hit =
        _accel->traverse_any(ray, {})
            .on_surface_candidate([&](compute::SurfaceCandidate &c) noexcept {
                auto hit = c.hit();
                auto bary = make_float3(1.f - hit.bary.x - hit.bary.y, hit.bary);
                auto it = interaction(hit.inst, hit.prim, bary, -ray->direction());
                $if(it->shape().maybe_non_opaque() & it->shape().has_surface()) {
                    PolymorphicCall<Surface::Closure> call;
                    _pipeline.surfaces().dispatch(it->shape().surface_tag(), [&](auto surface) noexcept {
                        $if(surface->maybe_non_opaque()) {
                            // TODO: pass the correct time
                            surface->closure(call, *it, _pipeline.spectrum()->sample(.5f), -ray->direction(), 1.f, 0.f);
                        };
                    });
                    auto u1 = xxhash32(as<uint3>(ray->origin()));
                    auto u2 = xxhash32(as<uint3>(ray->direction()));
                    auto u = xxhash32(make_uint2(u1, u2)) * 0x1p-32f;
                    call.execute([&](const Surface::Closure *closure) noexcept {
                        // apply opacity map
                        auto alpha_skip = def(false);
                        if (auto o = closure->opacity()) {
                            auto opacity = saturate(*o);
                            alpha_skip = u >= opacity;
                        }
                        $if(!alpha_skip) {
                            c.commit();
                        };
                    });
                }
                $else {
                    c.commit();
                };
            })
            .trace();
    return !rq_hit->miss();
}

luisa::shared_ptr<Interaction> Geometry::interaction(Expr<uint> inst_id, Expr<uint> prim_id,
                                                     Expr<float3> bary, Expr<float3> wo) const noexcept {
    auto shape = instance(inst_id);
    auto m = instance_to_world(inst_id);
    auto tri = triangle(shape, prim_id);
    auto attrib = shading_point(shape, tri, bary, m);
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

Shape::Handle Geometry::instance(Expr<uint> index) const noexcept {
    return Shape::Handle::decode(_instance_buffer->read(index));
}

Float4x4 Geometry::instance_to_world(Expr<uint> index) const noexcept {
    return _accel->instance_transform(index);
}

Var<Triangle> Geometry::triangle(const Shape::Handle &instance, Expr<uint> index) const noexcept {
    return _pipeline.buffer<Triangle>(instance.triangle_buffer_id()).read(index);
}

template<typename T>
[[nodiscard]] inline auto interpolate(Expr<float3> uvw,
                                      const T &v0,
                                      const T &v1,
                                      const T &v2) noexcept {
    return uvw.x * v0 + uvw.y * v1 + uvw.z * v2;
}

GeometryAttribute Geometry::geometry_point(const Shape::Handle &instance, const Var<Triangle> &triangle,
                                           const Var<float3> &bary, const Var<float4x4> &shape_to_world) const noexcept {
    auto v_buffer = instance.vertex_buffer_id();
    auto v0 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
    auto v1 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
    auto v2 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
    // object space
    auto p0 = v0->position();
    auto p1 = v1->position();
    auto p2 = v2->position();
    auto m = make_float3x3(shape_to_world);
    auto t = make_float3(shape_to_world[3]);
    // world space
    auto p = m * interpolate(bary, p0, p1, p2) + t;
    auto dp0 = p1 - p0;
    auto dp1 = p2 - p0;
    auto c = cross(m * dp0, m * dp1);
    auto area = length(c) * .5f;
    auto ng = normalize(c);
    return {.p = p, .n = ng, .area = area};
}

ShadingAttribute Geometry::shading_point(const Shape::Handle &instance, const Var<Triangle> &triangle,
                                         const Var<float3> &bary, const Var<float4x4> &shape_to_world) const noexcept {
    auto v_buffer = instance.vertex_buffer_id();
    auto v0 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
    auto v1 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
    auto v2 = _pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
    // object space
    auto p0_local = v0->position();
    auto p1_local = v1->position();
    auto p2_local = v2->position();
    auto ns_local = interpolate(bary, v0->normal(), v1->normal(), v2->normal());
    // compute dpdu and dpdv
    auto uv0 = v0->uv();
    auto uv1 = v1->uv();
    auto uv2 = v2->uv();
    auto duv0 = uv1 - uv0;
    auto duv1 = uv2 - uv0;
    auto det = duv0.x * duv1.y - duv0.y * duv1.x;
    auto inv_det = 1.f / det;
    auto dp0_local = p1_local - p0_local;
    auto dp1_local = p2_local - p0_local;
    auto dpdu_local = (dp0_local * duv1.y - dp1_local * duv0.y) * inv_det;
    auto dpdv_local = (dp1_local * duv0.x - dp0_local * duv1.x) * inv_det;
    // world space
    auto m = make_float3x3(shape_to_world);
    auto t = make_float3(shape_to_world[3]);
    auto p = m * interpolate(bary, p0_local, p1_local, p2_local) + t;
    auto c = cross(m * dp0_local, m * dp1_local);
    auto area = length(c) * .5f;
    auto ng = normalize(c);
    auto fallback_frame = Frame::make(ng);
    auto dpdu = ite(det == 0.f, fallback_frame.s(), m * dpdu_local);
    auto dpdv = ite(det == 0.f, fallback_frame.t(), m * dpdv_local);
    auto mn = transpose(inverse(m));
    auto ns = ite(instance.has_vertex_normal(), normalize(mn * ns_local), ng);
    auto uv = ite(instance.has_vertex_uv(), interpolate(bary, uv0, uv1, uv2), bary.yz());
    return {.g = {.p = p,
                  .n = ng,
                  .area = area},
            .ps = p,
            .ns = face_forward(ns, ng),
            .dpdu = dpdu,
            .dpdv = dpdv,
            .uv = uv};
}

}// namespace luisa::render

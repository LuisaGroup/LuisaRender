//
// Created by Mike on 2021/12/15.
//

#include <luisa-compute.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/scene.h>

#ifdef interface
#undef interface// Windows is really bad...
#endif

namespace luisa::render {

inline Pipeline::Pipeline(Device &device) noexcept
    : _device{device},
      _bindless_array{device.create_bindless_array(bindless_array_capacity)},
      _position_buffer_arena{luisa::make_unique<BufferArena>(
          device, vertex_buffer_arena_size_elements * sizeof(float3))},
      _attribute_buffer_arena{luisa::make_unique<BufferArena>(
          device, vertex_buffer_arena_size_elements * sizeof(Shape::VertexAttribute))},
      _general_buffer_arena{luisa::make_unique<BufferArena>(device, 16_mb)} {}

Pipeline::~Pipeline() noexcept = default;

void Pipeline::_build_geometry(
    CommandBuffer &command_buffer, luisa::span<const Shape *const> shapes,
    float init_time, AccelBuildHint hint) noexcept {

    _accel = _device.create_accel(hint);
    for (auto shape : shapes) { _process_shape(command_buffer, shape); }
    _instance_buffer = _device.create_buffer<Shape::Handle>(_instances.size());
    command_buffer << _instance_buffer.copy_from(_instances.data())
                   << _accel.build();// FIXME: adding commit() leads to wrong rendering, why?
}

void Pipeline::_process_shape(
    CommandBuffer &command_buffer, const Shape *shape,
    luisa::optional<bool> overridden_two_sided,
    const Surface *overridden_surface,
    const Light *overridden_light) noexcept {

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
                auto positions = shape->positions();
                auto attributes = shape->attributes();
                auto triangles = shape->triangles();
                if (positions.empty() || triangles.empty()) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION("Found mesh without vertices.");
                }
                if (positions.size() != attributes.size()) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION(
                        "Sizes of positions ({}) and "
                        "attributes ({}) mismatch.",
                        positions.size(), attributes.size());
                }
                auto hash = luisa::detail::xxh3_hash64(positions.data(), positions.size_bytes(), Hash64::default_seed);
                hash = luisa::detail::xxh3_hash64(attributes.data(), attributes.size_bytes(), hash);
                hash = luisa::detail::xxh3_hash64(triangles.data(), triangles.size_bytes(), hash);
                auto [cache_iter, non_existent] = _mesh_cache.try_emplace(hash, MeshGeometry{});
                if (!non_existent) { return cache_iter->second; }

                // create mesh
                auto position_buffer_view = _position_buffer_arena->allocate<float3>(positions.size());
                auto attribute_buffer_view = _attribute_buffer_arena->allocate<Shape::VertexAttribute>(attributes.size());
                if (position_buffer_view.offset() != attribute_buffer_view.offset()) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION("Position and attribute buffer offsets mismatch.");
                }
                auto index_offset = static_cast<uint>(position_buffer_view.offset());
                luisa::vector<Triangle> offset_triangles(triangles.size());
                std::transform(triangles.cbegin(), triangles.cend(), offset_triangles.begin(), [index_offset](auto t) noexcept {
                    return Triangle{t.i0 + index_offset, t.i1 + index_offset, t.i2 + index_offset};
                });
                auto triangle_buffer = create<Buffer<Triangle>>(triangles.size());
                command_buffer << position_buffer_view.copy_from(positions.data())
                               << attribute_buffer_view.copy_from(attributes.data())
                               << triangle_buffer->copy_from(offset_triangles.data())
                               << compute::commit();
                auto mesh = create<Mesh>(position_buffer_view.original(), *triangle_buffer, shape->build_hint());
                command_buffer << mesh->build()
                               << compute::commit();
                // compute alias table
                luisa::vector<float> triangle_areas(triangles.size());
                std::transform(triangles.cbegin(), triangles.cend(), triangle_areas.begin(), [positions](auto t) noexcept {
                    auto p0 = positions[t.i0];
                    auto p1 = positions[t.i1];
                    auto p2 = positions[t.i2];
                    return std::abs(length(cross(p1 - p0, p2 - p0)));
                });
                auto [alias_table, pdf] = create_alias_table(triangle_areas);
                auto alias_table_buffer_view = _general_buffer_arena->allocate<AliasEntry>(alias_table.size());
                auto pdf_buffer_view = _general_buffer_arena->allocate<float>(pdf.size());
                command_buffer << alias_table_buffer_view.copy_from(alias_table.data())
                               << pdf_buffer_view.copy_from(pdf.data())
                               << compute::commit();
                auto position_buffer_id = register_bindless(position_buffer_view.original());
                auto attribute_buffer_id = register_bindless(attribute_buffer_view.original());
                auto triangle_buffer_id = register_bindless(triangle_buffer->view());
                auto alias_buffer_id = register_bindless(alias_table_buffer_view);
                auto pdf_buffer_id = register_bindless(pdf_buffer_view);
                return cache_iter->second = {mesh, position_buffer_id};
            }();
            // assign mesh data
            MeshData mesh{};
            mesh.resource = mesh_geom.resource;
            mesh.geometry_buffer_id_base = mesh_geom.buffer_id_base;
            mesh.two_sided = shape->two_sided().value_or(false);
            iter = _meshes.emplace(shape, mesh).first;
        }
        auto mesh = iter->second;
        auto two_sided = overridden_two_sided.value_or(mesh.two_sided);
        auto instance_id = static_cast<uint>(_accel.size());
        auto [t_node, is_static] = _transform_tree.leaf(shape->transform());
        InstancedTransform inst_xform{t_node, instance_id};
        if (!is_static) { _dynamic_transforms.emplace_back(inst_xform); }
        auto object_to_world = inst_xform.matrix(_mean_time);
        _accel.emplace_back(*mesh.resource, object_to_world, true);

        // create instance
        auto properties = 0u;
        auto surface_tag = 0u;
        auto light_tag = 0u;
        if (two_sided) { properties |= Shape::property_flag_two_sided; }
        if (surface != nullptr && !surface->is_null()) {
            surface_tag = _process_surface(command_buffer, surface);
            properties |= Shape::property_flag_has_surface;
        }
        if (light != nullptr && !light->is_null()) {
            light_tag = _process_light(command_buffer, light);
            properties |= Shape::property_flag_has_light;
        }
        _instances.emplace_back(Shape::Handle::encode(
            mesh.geometry_buffer_id_base,
            properties, surface_tag, light_tag,
            mesh.resource->triangle_count()));
        if (properties & Shape::property_flag_has_light) {
            _instanced_lights.emplace_back(Light::Handle{
                .instance_id = instance_id,
                .light_tag = light_tag});
        }
    } else {
        _transform_tree.push(shape->transform());
        for (auto child : shape->children()) {
            _process_shape(command_buffer, child, shape->two_sided(), surface, light);
        }
        _transform_tree.pop(shape->transform());
    }
}

uint Pipeline::_process_surface(CommandBuffer &command_buffer, const Surface *surface) noexcept {
    auto [iter, not_existent] = _surface_tags.try_emplace(surface, 0u);
    if (not_existent) {
        iter->second = static_cast<uint>(_surfaces.size());
        _surfaces.emplace_back(surface->build(*this, command_buffer));
    }
    return iter->second;
}

uint Pipeline::_process_light(CommandBuffer &command_buffer, const Light *light) noexcept {
    auto [iter, not_existent] = _light_tags.try_emplace(light, 0u);
    if (not_existent) {
        iter->second = static_cast<uint>(_lights.size());
        _lights.emplace_back(light->build(*this, command_buffer));
    }
    return iter->second;
}

luisa::unique_ptr<Pipeline> Pipeline::create(Device &device, Stream &stream, const Scene &scene) noexcept {
    ThreadPool::global().synchronize();
    auto pipeline = luisa::make_unique<Pipeline>(device);
    pipeline->_cameras.reserve(scene.cameras().size());
    pipeline->_films.reserve(scene.cameras().size());
    pipeline->_filters.reserve(scene.cameras().size());
    auto command_buffer = stream.command_buffer();
    auto rgb2spec_t0 = pipeline->create<Volume<float>>(PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::resolution));
    auto rgb2spec_t1 = pipeline->create<Volume<float>>(PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::resolution));
    auto rgb2spec_t2 = pipeline->create<Volume<float>>(PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::resolution));
    RGB2SpectrumTable::srgb().encode(
        command_buffer,
        rgb2spec_t0->view(0u),
        rgb2spec_t1->view(0u),
        rgb2spec_t2->view(0u));
    pipeline->_rgb2spec_index = pipeline->register_bindless(*rgb2spec_t0, TextureSampler::linear_point_zero());
    static_cast<void>(pipeline->register_bindless(*rgb2spec_t1, TextureSampler::linear_point_zero()));
    static_cast<void>(pipeline->register_bindless(*rgb2spec_t2, TextureSampler::linear_point_zero()));
    auto mean_time = 0.0;
    for (auto camera : scene.cameras()) {
        pipeline->_cameras.emplace_back(camera->build(*pipeline, command_buffer));
        pipeline->_films.emplace_back(camera->film()->build(*pipeline, command_buffer));
        pipeline->_filters.emplace_back(camera->filter()->build(*pipeline, command_buffer));
        mean_time += (camera->shutter_span().x + camera->shutter_span().y) * 0.5f;
    }
    mean_time *= 1.0 / static_cast<double>(scene.cameras().size());
    pipeline->_mean_time = static_cast<float>(mean_time);
    pipeline->_build_geometry(command_buffer, scene.shapes(), pipeline->_mean_time, AccelBuildHint::FAST_TRACE);
    pipeline->_integrator = scene.integrator()->build(*pipeline, command_buffer);
    pipeline->_sampler = scene.integrator()->sampler()->build(*pipeline, command_buffer);
    if (auto env = scene.environment(); env != nullptr && !env->is_black()) {
        pipeline->_environment = env->build(*pipeline, command_buffer);
    }
    if (pipeline->_lights.empty()) [[unlikely]] {
        if (pipeline->_environment == nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "No lights or environment found in the scene.");
        }
    } else {
        pipeline->_light_sampler = scene.integrator()->light_sampler()->build(*pipeline, command_buffer);
    }
    command_buffer << pipeline->_bindless_array.update()
                   << compute::commit();
    return pipeline;
}

bool Pipeline::update_geometry(CommandBuffer &command_buffer, float time) noexcept {
    // TODO: support deformable meshes
    if (_dynamic_transforms.empty()) { return false; }
    if (_dynamic_transforms.size() < 128u) {
        for (auto t : _dynamic_transforms) {
            _accel.set_transform(
                t.instance_id(),
                t.matrix(time));
        }
    } else {
        ThreadPool::global().parallel(
            _dynamic_transforms.size(),
            [this, time](auto i) noexcept {
                auto t = _dynamic_transforms[i];
                _accel.set_transform(
                    t.instance_id(),
                    t.matrix(time));
            });
        ThreadPool::global().synchronize();
    }
    command_buffer << _accel.update()
                   << luisa::compute::commit();
    return true;
}

void Pipeline::render(Stream &stream) noexcept {
    _integrator->render(stream);
}

std::tuple<Camera::Instance *, Film::Instance *, Filter::Instance *> Pipeline::camera(size_t i) noexcept {
    return std::make_tuple(_cameras[i].get(), _films[i].get(), _filters[i].get());
}

std::tuple<const Camera::Instance *, const Film::Instance *, const Filter::Instance *> Pipeline::camera(size_t i) const noexcept {
    return std::make_tuple(_cameras[i].get(), _films[i].get(), _filters[i].get());
}

std::pair<Var<Shape::Handle>, Var<float4x4>> Pipeline::instance(Expr<uint> i) const noexcept {
    auto instance = _instance_buffer.read(i);
    auto transform = _accel.instance_to_world(i);
    return std::make_pair(std::move(instance), std::move(transform));
}

Var<Triangle> Pipeline::triangle(const Var<Shape::Handle> &instance, Expr<uint> i) const noexcept {
    return buffer<Triangle>(instance->triangle_buffer_id()).read(i);
}

std::tuple<Var<float3>, Var<float3>, Var<float>> Pipeline::surface_point_geometry(
    const Var<Shape::Handle> &instance, const Var<float4x4> &shape_to_world,
    const Var<Triangle> &triangle, const Var<float3> &uvw) const noexcept {

    auto world = [&m = shape_to_world](auto &&p) noexcept {
        return make_float3(m * make_float4(std::forward<decltype(p)>(p), 1.0f));
    };
    auto p_buffer = instance->position_buffer_id();
    auto p0 = world(buffer<float3>(p_buffer).read(triangle.i0));
    auto p1 = world(buffer<float3>(p_buffer).read(triangle.i1));
    auto p2 = world(buffer<float3>(p_buffer).read(triangle.i2));
    auto p = uvw.x * p0 + uvw.y * p1 + uvw.z * p2;
    auto c = cross(p1 - p0, p2 - p0);
    auto area = 0.5f * length(c);
    auto ng = normalize(c);
    return std::make_tuple(std::move(p), std::move(ng), std::move(area));
}

std::tuple<Var<float3>, Var<float3>, Var<float2>> Pipeline::surface_point_attributes(
    const Var<Shape::Handle> &instance, const Var<float3x3> &shape_to_world_normal, const Var<Triangle> &triangle, const Var<float3> &uvw) const noexcept {
    auto a0 = buffer<Shape::VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i0);
    auto a1 = buffer<Shape::VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i1);
    auto a2 = buffer<Shape::VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i2);
    auto interpolate = [&](auto &&a, auto &&b, auto &&c) noexcept {
        return uvw.x * std::forward<decltype(a)>(a) +
               uvw.y * std::forward<decltype(b)>(b) +
               uvw.z * std::forward<decltype(c)>(c);
    };
    auto normal = normalize(shape_to_world_normal * interpolate(a0->normal(), a1->normal(), a2->normal()));
    auto tangent = normalize(shape_to_world_normal * interpolate(a0->tangent(), a1->tangent(), a2->tangent()));
    auto uv = interpolate(a0->uv(), a1->uv(), a2->uv());
    return std::make_tuple(std::move(normal), std::move(tangent), std::move(uv));
}

Var<Hit> Pipeline::trace_closest(const Var<Ray> &ray) const noexcept { return _accel.trace_closest(ray); }
Var<bool> Pipeline::trace_any(const Var<Ray> &ray) const noexcept { return _accel.trace_any(ray); }

luisa::unique_ptr<Interaction> Pipeline::interaction(const Var<Ray> &ray, const Var<Hit> &hit) const noexcept {
    using namespace luisa::compute;
    Interaction it;
    $if(hit->miss()) {
        it = Interaction{-ray->direction()};
    }
    $else {
        auto [shape, shape_to_world] = instance(hit.inst);
        auto shape_to_world_normal = transpose(inverse(make_float3x3(shape_to_world)));
        auto tri = triangle(shape, hit.prim);
        auto [p, ng, area] = surface_point_geometry(
            shape, shape_to_world, tri,
            make_float3(1.0f - hit.bary.x - hit.bary.y, hit.bary));
        auto [ns, t, uv] = surface_point_attributes(
            shape, shape_to_world_normal, tri,
            make_float3(1.0f - hit.bary.x - hit.bary.y, hit.bary));
        auto wo = -ray->direction();
        auto &shape_ref = shape;
        auto &uv_ref = uv;
        it = Interaction{
            std::move(shape), hit.inst, hit.prim,
            area, p, wo, ng, uv, ns, t};
    };
    return luisa::make_unique<Interaction>(std::move(it));
}

RGBAlbedoSpectrum Pipeline::srgb_albedo_spectrum(Expr<float3> rgb) const noexcept {
    auto rsp = RGB2SpectrumTable::srgb().decode_albedo(
        Expr{_bindless_array}, _rgb2spec_index, rgb);
    return RGBAlbedoSpectrum{std::move(rsp)};
}

RGBUnboundSpectrum Pipeline::srgb_unbound_spectrum(Expr<float3> rgb) const noexcept {
    auto [rsp, scale] = RGB2SpectrumTable::srgb().decode_unbound(
        Expr{_bindless_array}, _rgb2spec_index, rgb);
    return {std::move(rsp), std::move(scale)};
}

RGBIlluminantSpectrum Pipeline::srgb_illuminant_spectrum(Expr<float3> rgb) const noexcept {
    auto [rsp, scale] = RGB2SpectrumTable::srgb().decode_unbound(
        Expr{_bindless_array}, _rgb2spec_index, rgb);
    return {std::move(rsp), std::move(scale), DenselySampledSpectrum::cie_illum_d65()};
}

const Texture::Instance *Pipeline::build_texture(CommandBuffer &command_buffer, const Texture *texture) noexcept {
    if (texture == nullptr) { return nullptr; }
    auto [iter, not_exists] = _textures.try_emplace(texture, nullptr);
    if (not_exists) { iter->second = texture->build(*this, command_buffer); }
    return iter->second.get();
}

void Pipeline::dynamic_dispatch_surface(
    Expr<uint> tag, const function<void(const Surface::Instance *)> &f) const noexcept {
    if (!_surfaces.empty()) [[likely]] {
        $switch(tag) {
            for (auto i = 0u; i < _surfaces.size(); i++) {
                $case(i) { f(_surfaces[i].get()); };
            }
            $default { compute::unreachable(); };
        };
    }
}

void Pipeline::dynamic_dispatch_light(
    Expr<uint> tag, const function<void(const Light::Instance *)> &f) const noexcept {
    if (!_lights.empty()) [[likely]] {
        $switch(tag) {
            for (auto i = 0u; i < _lights.size(); i++) {
                $case(i) { f(_lights[i].get()); };
            }
            $default { compute::unreachable(); };
        };
    }
}

}// namespace luisa::render

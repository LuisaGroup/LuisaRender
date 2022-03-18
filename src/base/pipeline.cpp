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
      _general_buffer_arena{luisa::make_unique<BufferArena>(device, 16_mb)},
      _printer{compute::Printer{device}} {}

Pipeline::~Pipeline() noexcept = default;

void Pipeline::_build_geometry(
    CommandBuffer &command_buffer, luisa::span<const Shape *const> shapes,
    float init_time, AccelBuildHint hint) noexcept {

    _accel = _device.create_accel(hint);
    for (auto shape : shapes) { _process_shape(command_buffer, shape); }
    _instance_buffer = _device.create_buffer<Shape::Handle>(_instances.size());
    command_buffer << _instance_buffer.copy_from(_instances.data())
                   << _accel.build();
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
                auto vertices = shape->vertices();
                auto triangles = shape->triangles();
                if (vertices.empty() || triangles.empty()) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION("Found mesh without vertices.");
                }
                auto hash = luisa::detail::xxh3_hash64(vertices.data(), vertices.size_bytes(), Hash64::default_seed);
                hash = luisa::detail::xxh3_hash64(triangles.data(), triangles.size_bytes(), hash);
                auto [cache_iter, non_existent] = _mesh_cache.try_emplace(hash, MeshGeometry{});
                if (!non_existent) { return cache_iter->second; }

                // create mesh
                auto vertex_buffer = create<Buffer<Shape::Vertex>>(vertices.size());
                auto triangle_buffer = create<Buffer<Triangle>>(triangles.size());
                auto mesh = create<Mesh>(*vertex_buffer, *triangle_buffer, shape->build_hint());
                command_buffer << vertex_buffer->copy_from(vertices.data())
                               << triangle_buffer->copy_from(triangles.data())
                               << mesh->build()
                               << compute::commit();
                // compute alias table
                luisa::vector<float> triangle_areas(triangles.size());
                std::transform(triangles.cbegin(), triangles.cend(), triangle_areas.begin(), [vertices](auto t) noexcept {
                    auto p0 = vertices[t.i0].pos;
                    auto p1 = vertices[t.i1].pos;
                    auto p2 = vertices[t.i2].pos;
                    return std::abs(length(cross(p1 - p0, p2 - p0)));
                });
                auto [alias_table, pdf] = create_alias_table(triangle_areas);
                auto alias_table_buffer_view = _general_buffer_arena->allocate<AliasEntry>(alias_table.size());
                auto pdf_buffer_view = _general_buffer_arena->allocate<float>(pdf.size());
                command_buffer << alias_table_buffer_view.copy_from(alias_table.data())
                               << pdf_buffer_view.copy_from(pdf.data())
                               << compute::commit();
                auto vertex_buffer_id = register_bindless(vertex_buffer->view());
                auto triangle_buffer_id = register_bindless(triangle_buffer->view());
                auto alias_buffer_id = register_bindless(alias_table_buffer_view);
                auto pdf_buffer_id = register_bindless(pdf_buffer_view);
                LUISA_ASSERT(triangle_buffer_id - vertex_buffer_id == Shape::Handle::triangle_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(alias_buffer_id - vertex_buffer_id == Shape::Handle::alias_table_buffer_id_offset, "Invalid.");
                LUISA_ASSERT(pdf_buffer_id - vertex_buffer_id == Shape::Handle::pdf_buffer_id_offset, "Invalid.");
                return cache_iter->second = {mesh, vertex_buffer_id};
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
    if (scene.integrator()->differentiable()) {
        pipeline->_differentiation =
            luisa::make_unique<Differentiation>(*pipeline);
    }
    pipeline->_cameras.reserve(scene.cameras().size());
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
    if (pipeline->_lights.empty() && pipeline->_environment == nullptr) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights or environment found in the scene.");
    } else {
        pipeline->_light_sampler = scene.integrator()->light_sampler()->build(*pipeline, command_buffer);
    }
    command_buffer << pipeline->_bindless_array.update();
    if (auto &&diff = pipeline->_differentiation) {
        diff->materialize(command_buffer);
    }
    command_buffer << compute::synchronize();
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
    command_buffer << _accel.update();
    if (_light_sampler) { _light_sampler->update(command_buffer, time); }
    command_buffer.commit();
    return true;
}

void Pipeline::render(Stream &stream) noexcept {
    _integrator->render(stream);
}

Var<Shape::Handle> Pipeline::instance(Expr<uint> i) const noexcept {
    return _instance_buffer.read(i);
}

Float4x4 Pipeline::instance_to_world(Expr<uint> i) const noexcept {
    return _accel.instance_to_world(i);
}

Var<Triangle> Pipeline::triangle(const Var<Shape::Handle> &instance, Expr<uint> i) const noexcept {
    return buffer<Triangle>(instance->triangle_buffer_id()).read(i);
}

ShadingAttribute Pipeline::shading_point(
    const Var<Shape::Handle> &instance, const Var<Triangle> &triangle, const Var<float3> &uvw,
    const Var<float4x4> &shape_to_world, const Var<float3x3> &shape_to_world_normal) const noexcept {
    auto v_buffer = instance->vertex_buffer_id();
    auto v0 = buffer<Shape::Vertex>(v_buffer).read(triangle.i0);
    auto v1 = buffer<Shape::Vertex>(v_buffer).read(triangle.i1);
    auto v2 = buffer<Shape::Vertex>(v_buffer).read(triangle.i2);
    auto p0 = make_float3(shape_to_world * make_float4(v0->position(), 1.f));
    auto p1 = make_float3(shape_to_world * make_float4(v1->position(), 1.f));
    auto p2 = make_float3(shape_to_world * make_float4(v2->position(), 1.f));
    auto p = uvw.x * p0 + uvw.y * p1 + uvw.z * p2;
    auto c = cross(p1 - p0, p2 - p0);
    auto area = 0.5f * length(c);
    auto ng = normalize(c);
    auto uv = uvw.x * v0->uv() + uvw.y * v1->uv() + uvw.z * v2->uv();
    auto ns_local = uvw.x * v0->normal() + uvw.y * v1->normal() + uvw.z * v2->normal();
    auto tangent_local = uvw.x * v0->tangent() + uvw.y * v1->tangent() + uvw.z * v2->tangent();
    auto ns = normalize(shape_to_world_normal * ns_local);
    auto tangent = normalize(shape_to_world_normal * tangent_local);
    return {.p = p, .ng = ng, .ns = ns, .tangent = tangent, .uv = uv, .area = area};
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

Differentiation &Pipeline::differentiation() noexcept {
    LUISA_ASSERT(_differentiation, "Differentiation is not constructed.");
    return *_differentiation;
}

const Differentiation &Pipeline::differentiation() const noexcept {
    LUISA_ASSERT(_differentiation, "Differentiation is not constructed.");
    return *_differentiation;
}

}// namespace luisa::render

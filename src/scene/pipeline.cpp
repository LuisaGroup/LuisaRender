//
// Created by Mike on 2021/12/15.
//

#include <luisa-compute.h>
#include <util/sampling.h>
#include <scene/pipeline.h>

namespace luisa::render {

inline Pipeline::Pipeline(Device &device) noexcept
    : _device{device},
      _bindless_array{device.create_bindless_array(bindless_array_capacity)},
      _position_buffer_arena{luisa::make_unique<BufferArena>(
          device, vertex_buffer_arena_size_elements * sizeof(float3))},
      _attribute_buffer_arena{luisa::make_unique<BufferArena>(
          device, vertex_buffer_arena_size_elements * sizeof(VertexAttribute))},
      _general_buffer_arena{luisa::make_unique<BufferArena>(device, 16_mb)} {}

Pipeline::~Pipeline() noexcept = default;

void Pipeline::_build_geometry(CommandBuffer &command_buffer, luisa::span<const Shape *const> shapes, float init_time, AccelBuildHint hint) noexcept {
    _accel = _device.create_accel(hint);
    for (auto shape : shapes) { _process_shape(command_buffer, shape); }
    _instance_buffer = _device.create_buffer<InstancedShape>(_instances.size());
    command_buffer << _bindless_array.update()
                   << _instance_buffer.copy_from(_instances.data())
                   << _accel.build();
}

void Pipeline::_process_shape(
    CommandBuffer &command_buffer, const Shape *shape,
    luisa::optional<bool> overridden_two_sided,
    const Material *overridden_material,
    const Light *overridden_light) noexcept {

    auto material = overridden_material == nullptr ? shape->material() : overridden_material;
    auto light = overridden_light == nullptr ? shape->light() : overridden_light;

    if (shape->is_mesh()) {
        if (shape->deformable()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Deformable meshes are not yet supported.");
        }
        auto iter = _meshes.find(shape);
        if (iter == _meshes.end()) {
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
            MeshData mesh{};
            // create mesh
            auto position_buffer_view = _position_buffer_arena->allocate<float3>(positions.size());
            auto attribute_buffer_view = _attribute_buffer_arena->allocate<VertexAttribute>(attributes.size());
            if (position_buffer_view.offset() != attribute_buffer_view.offset()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Position and attribute buffer offsets mismatch.");
            }
            auto index_offset = static_cast<uint>(position_buffer_view.offset());
            luisa::vector<Triangle> offset_triangles(triangles.size());
            std::transform(triangles.cbegin(), triangles.cend(), offset_triangles.begin(), [index_offset](auto t) noexcept {
                return Triangle{t.i0 + index_offset, t.i1 + index_offset, t.i2 + index_offset};
            });
            auto triangle_buffer = create<Buffer<Triangle>>(triangles.size());
            mesh.resource = create<Mesh>(position_buffer_view.original(), *triangle_buffer, shape->build_hint());
            command_buffer << position_buffer_view.copy_from(positions.data())
                           << attribute_buffer_view.copy_from(attributes.data())
                           << triangle_buffer->copy_from(offset_triangles.data())
                           << mesh.resource->build()
                           << luisa::compute::commit();
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
                           << luisa::compute::commit();

            // assign mesh data
            auto position_buffer_id = register_bindless(position_buffer_view.original());
            auto attribute_buffer_id = register_bindless(attribute_buffer_view.original());
            auto triangle_buffer_id = register_bindless(triangle_buffer->view());
            auto alias_buffer_id = register_bindless(alias_table_buffer_view);
            auto pdf_buffer_id = register_bindless(pdf_buffer_view);
            mesh.buffer_id_base = position_buffer_id;
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
        if (shape->is_virtual()) {
            auto scaling = make_float4x4(make_float3x3(0.0f));
            _accel.emplace_back(*mesh.resource, object_to_world * scaling, false);
        } else {
            _accel.emplace_back(*mesh.resource, object_to_world, true);
        }

        // create instance
        InstancedShape instance{};
        instance.buffer_id_base = mesh.buffer_id_base;
        if (two_sided) { instance.properties |= Shape::property_flag_two_sided; }
        if (material != nullptr && !material->is_black()) {
            if (shape->is_virtual()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Materials will be ignored on virtual shapes.");
            } else {
                auto m = _process_material(command_buffer, instance_id, shape, material);
                instance.properties |= Shape::property_flag_has_material;
                instance.material_buffer_id_and_tag = InstancedShape::encode_material_buffer_id_and_tag(m.buffer_id, m.tag);
            }
        }
        if (light != nullptr && !light->is_black()) {
            if (shape->is_virtual() != light->is_virtual()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Non-virtual lights will be ignored on "
                    "virtual shapes and vise versa.");
            } else {
                auto l = _process_light(command_buffer, instance_id, shape, light);
                instance.properties |= Shape::property_flag_has_light;
                instance.light_buffer_id_and_tag = InstancedShape::encode_light_buffer_id_and_tag(l.buffer_id, l.tag);
            }
        }
        _instances.emplace_back(instance);
    } else {
        //        if (!shape->is_virtual()) {
        _transform_tree.push(shape->transform());
        for (auto child : shape->children()) {
            _process_shape(command_buffer, child, shape->two_sided(), material, light);
        }
        _transform_tree.pop(shape->transform());
        //        }
    }
}

Pipeline::MaterialData Pipeline::_process_material(CommandBuffer &command_buffer, uint instance_id, const Shape *shape, const Material *material) noexcept {
    if (auto iter = _materials.find(material); iter != _materials.cend()) { return iter->second; }
    auto tag = [this, material] {
        luisa::string impl_type{material->impl_type()};
        if (auto iter = _material_tags.find(impl_type);
            iter != _material_tags.cend()) { return iter->second; }
        auto t = static_cast<uint32_t>(_material_interfaces.size());
        if (t > InstancedShape::material_tag_mask) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Too many material tags.");
        }
        _material_interfaces.emplace_back(material);
        _material_tags.emplace(std::move(impl_type), t);
        return t;
    }();
    auto buffer_id = material->encode(*this, command_buffer, instance_id, shape);
    return _materials.emplace(material, MaterialData{shape, instance_id, buffer_id, tag}).first->second;
}

Pipeline::LightData Pipeline::_process_light(CommandBuffer &command_buffer, uint instance_id, const Shape *shape, const Light *light) noexcept {
    if (auto iter = _lights.find(light); iter != _lights.cend()) { return iter->second; }
    auto tag = [this, light] {
        luisa::string impl_type{light->impl_type()};
        if (auto iter = _light_tags.find(impl_type);
            iter != _light_tags.cend()) { return iter->second; }
        auto t = static_cast<uint32_t>(_light_interfaces.size());
        if (t > InstancedShape::light_tag_mask) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Too many light tags.");
        }
        _light_interfaces.emplace_back(light);
        _light_tags.emplace(std::move(impl_type), t);
        return t;
    }();
    auto buffer_id = light->encode(*this, command_buffer, instance_id, shape);
    return _lights.emplace(light, LightData{shape, instance_id, buffer_id, tag}).first->second;
}

luisa::unique_ptr<Pipeline> Pipeline::create(Device &device, Stream &stream, const Scene &scene) noexcept {
    auto pipeline = luisa::make_unique<Pipeline>(device);
    pipeline->_cameras.reserve(scene.cameras().size());
    pipeline->_films.reserve(scene.cameras().size());
    pipeline->_filters.reserve(scene.cameras().size());
    auto command_buffer = stream.command_buffer();
    {
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
    }
    command_buffer.commit();
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

std::pair<Var<InstancedShape>, Var<float4x4>> Pipeline::instance(Expr<uint> i) const noexcept {
    auto instance = _instance_buffer.read(i);
    auto transform = _accel.instance_to_world(i);
    return std::make_pair(std::move(instance), std::move(transform));
}

Var<Triangle> Pipeline::triangle(const Var<InstancedShape> &instance, Expr<uint> i) const noexcept {
    return buffer<Triangle>(instance->triangle_buffer_id()).read(i);
}

std::tuple<Var<float3>, Var<float3>, Var<float>> Pipeline::surface_point_geometry(
    const Var<InstancedShape> &instance, const Var<float4x4> &shape_to_world,
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
    const Var<InstancedShape> &instance, const Var<float3x3> &shape_to_world_normal, const Var<Triangle> &triangle, const Var<float3> &uvw) const noexcept {
    auto a0 = buffer<VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i0);
    auto a1 = buffer<VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i1);
    auto a2 = buffer<VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i2);
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
        it = Interaction{std::move(shape), hit.prim, area, p, wo, ng, uv, ns, t};
    };
    return luisa::make_unique<Interaction>(std::move(it));
}

luisa::unique_ptr<Material::Closure> Pipeline::decode_material(uint tag, const Interaction &it) const noexcept {
    if (tag > _material_interfaces.size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid material tag: {}.", tag);
    }
    return _material_interfaces[tag]->decode(*this, it);
}

void Pipeline::decode_material(Expr<uint> tag, const Interaction &it, const luisa::function<void(const Material::Closure &)> &func) const noexcept {
    $switch(tag) {
        for (auto i = 0u; i < _material_interfaces.size(); i++) {
            $case(i) { func(*decode_material(i, it)); };
        }
    };
}

luisa::unique_ptr<Light::Closure> Pipeline::decode_light(uint tag) const noexcept {
    if (tag > _light_interfaces.size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid light tag: {}.", tag);
    }
    return _light_interfaces[tag]->decode(*this);
}

void Pipeline::decode_light(Expr<uint> tag, const function<void(const Light::Closure &)> &func) const noexcept {
    $switch(tag) {
        for (auto i = 0u; i < _light_interfaces.size(); i++) {
            $case(i) { func(*decode_light(i)); };
        }
    };
}

}// namespace luisa::render

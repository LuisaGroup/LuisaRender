//
// Created by Mike on 2021/12/15.
//

#include <luisa-compute.h>
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
    auto transform_builder = TransformTree::builder(init_time);
    for (auto shape : shapes) {
        _process_shape(command_buffer, transform_builder, shape);
    }
    _transform_tree = transform_builder.build();
    _instance_buffer = _device.create_buffer<InstancedShape>(_instances.size());
    command_buffer << _bindless_array.update()
                   << _instance_buffer.copy_from(_instances.data())
                   << _accel.build();
}

void Pipeline::_process_shape(
    CommandBuffer &command_buffer, TransformTree::Builder &transform_builder, const Shape *shape,
    luisa::optional<bool> overridden_two_sided, const Material *overridden_material, const Light *overridden_light) noexcept {

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
            // assign mesh data
            auto position_buffer_id = register_bindless(position_buffer_view.original());
            auto attribute_buffer_id = register_bindless(attribute_buffer_view.original());
            auto triangle_buffer_id = register_bindless(triangle_buffer->view());
            mesh.buffer_id_base = position_buffer_id;
            mesh.two_sided = shape->two_sided().value_or(false);
            iter = _meshes.emplace(shape, mesh).first;
        }
        auto mesh = iter->second;
        auto two_sided = overridden_two_sided.value_or(mesh.two_sided);
        auto [m, m_flags] = _process_material(command_buffer, shape, material);
        auto [l, l_flags] = _process_light(command_buffer, shape, light);

        // create instance
        InstancedShape instance{};
        instance.buffer_id_base = mesh.buffer_id_base;
        instance.properties = InstancedShape::encode_property_flags(
            two_sided ? Shape::property_flag_two_sided : 0u, m_flags, l_flags);
        instance.material_buffer_id_and_tag = m;
        instance.light_buffer_id_and_tag = l;
        // add instance
        auto object_to_world = transform_builder.leaf(shape->transform(), _accel.size());
        _accel.emplace_back(*mesh.resource, object_to_world, true);
        _instances.emplace_back(instance);
    } else {
        if (shape->transform() != nullptr) { transform_builder.push(shape->transform()); }
        for (auto child : shape->children()) { _process_shape(command_buffer, transform_builder, child, shape->two_sided(), material, light); }
        if (shape->transform() != nullptr) { transform_builder.pop(); }
    }
}

std::pair<uint, uint> Pipeline::_process_material(CommandBuffer &command_buffer, const Shape *shape, const Material *material) noexcept {
    if (material == nullptr) { return {~0u, Material::property_flag_black}; }
    if (auto iter = _materials.find(material); iter != _materials.cend()) {
        auto [_, buffer_id_and_tag, flags] = iter->second;
        return std::make_pair(buffer_id_and_tag, flags);
    }
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
    auto buffer_id_and_tag = InstancedShape::encode_material_buffer_id_and_tag(
        material->encode(*this, command_buffer, shape), tag);
    auto flags = material->property_flags();
    _materials.emplace(material, std::make_tuple(shape, buffer_id_and_tag, flags));
    return std::make_pair(buffer_id_and_tag, flags);
}

std::pair<uint, uint> Pipeline::_process_light(CommandBuffer &command_buffer, const Shape *shape, const Light *light) noexcept {
    if (light == nullptr) { return std::make_pair(~0u, Light::property_flag_black); }
    if (auto iter = _lights.find(light); iter != _lights.cend()) {
        auto [_, buffer_id_and_tag, flags] = iter->second;
        return std::make_pair(buffer_id_and_tag, flags);
    }
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
    auto buffer_id_and_tag = InstancedShape::encode_light_buffer_id_and_tag(
        light->encode(*this, command_buffer, shape), tag);
    auto flags = light->property_flags();
    _lights.emplace(light, std::make_tuple(shape, buffer_id_and_tag, flags));
    return std::make_pair(buffer_id_and_tag, flags);
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
            mean_time += (camera->time_span().x + camera->time_span().y) * 0.5f;
        }
        mean_time *= 1.0 / static_cast<double>(scene.cameras().size());
        pipeline->_build_geometry(command_buffer, scene.shapes(), static_cast<float>(mean_time), AccelBuildHint::FAST_TRACE);
        pipeline->_integrator = scene.integrator()->build(*pipeline, command_buffer);
        pipeline->_sampler = scene.integrator()->sampler()->build(*pipeline, command_buffer);
        pipeline->_light_sampler = scene.integrator()->light_sampler()->build(*pipeline, command_buffer);
    }
    command_buffer.commit();
    return pipeline;
}

void Pipeline::update_geometry(CommandBuffer &command_buffer, float time) noexcept {
    // TODO: support deformable meshes
    if (!_transform_tree.is_static()) {
        _transform_tree.update(_accel, time);
        command_buffer << _accel.update();
    }
}

void Pipeline::render(Stream &stream) noexcept {
    _integrator->render(stream, *this);
}

std::tuple<Camera::Instance *, Film::Instance *, Filter::Instance *> Pipeline::camera(size_t i) noexcept {
    return std::make_tuple(_cameras[i].get(), _films[i].get(), _filters[i].get());
}

std::tuple<const Camera::Instance *, const Film::Instance *, const Filter::Instance *> Pipeline::camera(size_t i) const noexcept {
    return std::make_tuple(_cameras[i].get(), _films[i].get(), _filters[i].get());
}

std::pair<Var<InstancedShape>, Var<float4x4>> Pipeline::instance(const Var<Hit> &hit) const noexcept {
    auto instance = _instance_buffer.read(hit.inst);
    auto transform = _accel.instance_to_world(hit.inst);
    return std::make_pair(std::move(instance), std::move(transform));
}

Var<Triangle> Pipeline::triangle(const Var<InstancedShape> &instance, const Var<Hit> &hit) const noexcept {
    return buffer<Triangle>(instance->triangle_buffer_id()).read(hit.prim);
}

std::pair<Var<float3>, Var<float3>> Pipeline::vertex(const Var<InstancedShape> &instance, const Var<float4x4> &shape_to_world, const Var<float3x3> &shape_to_world_normal, const Var<Triangle> &triangle, const Var<Hit> &hit) const noexcept {
    auto p0 = buffer<float3>(instance->position_buffer_id()).read(triangle.i0);
    auto p1 = buffer<float3>(instance->position_buffer_id()).read(triangle.i1);
    auto p2 = buffer<float3>(instance->position_buffer_id()).read(triangle.i2);
    auto p = make_float3(shape_to_world * make_float4(hit->interpolate(p0, p1, p2), 1.0f));
    auto ng = normalize(shape_to_world_normal * cross(p1 - p0, p2 - p0));
    return std::make_pair(std::move(p), std::move(ng));
}

std::tuple<Var<float3>, Var<float3>, Var<float2>> Pipeline::vertex_attributes(
    const Var<InstancedShape> &instance, const Var<float3x3> &shape_to_world_normal, const Var<Triangle> &triangle, const Var<Hit> &hit) const noexcept {
    auto a0 = buffer<VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i0);
    auto a1 = buffer<VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i1);
    auto a2 = buffer<VertexAttribute>(instance->attribute_buffer_id()).read(triangle.i2);
    auto normal = normalize(shape_to_world_normal * hit->interpolate(a0->normal(), a1->normal(), a2->normal()));
    auto tangent = normalize(shape_to_world_normal * hit->interpolate(a0->tangent(), a1->tangent(), a2->tangent()));
    auto uv = hit->interpolate(a0->uv(), a1->uv(), a2->uv());
    return std::make_tuple(std::move(normal), std::move(tangent), std::move(uv));
}

Var<Hit> Pipeline::trace_closest(const Var<Ray> &ray) const noexcept { return _accel.trace_closest(ray); }
Var<bool> Pipeline::trace_any(const Var<Ray> &ray) const noexcept { return _accel.trace_any(ray); }

luisa::unique_ptr<Interaction> Pipeline::interaction(const Var<Ray> &ray, const Var<Hit> &hit) const noexcept {
    using namespace luisa::compute;
    Interaction it;
    $if(!hit->miss()) {
        auto [shape, shape_to_world] = instance(hit);
        auto shape_to_world_normal = transpose(inverse(make_float3x3(shape_to_world)));
        auto tri = triangle(shape, hit);
        auto [p, ng] = vertex(shape, shape_to_world, shape_to_world_normal, tri, hit);
        auto [ns, t, uv] = vertex_attributes(shape, shape_to_world_normal, tri, hit);
        auto wo = -def<float3>(ray.direction);
        auto two_sided = shape->two_sided();
        it = Interaction{
            std::move(shape), p, wo,
            ite(two_sided & (dot(ng, wo) < 0.0f), -ng, ng), uv,
            ite(two_sided & (dot(ns, wo) < 0.0f), -ns, ns), t};
    };
    return luisa::make_unique<Interaction>(std::move(it));
}

luisa::unique_ptr<Material::Closure> Pipeline::decode_material(uint tag, const Interaction &it) const noexcept {
    if (tag > _material_interfaces.size()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid material tag: {}.", tag);
    }
    return _material_interfaces[tag]->decode(*this, it);
}

void Pipeline::decode_material(const Interaction &it, const luisa::function<void(const Material::Closure &)> &func) const noexcept {
    $switch(it.shape()->material_tag()) {
        for (auto tag = 0u; tag < _material_tags.size(); tag++) {
            $case(tag) {
                auto closure = decode_material(tag, it);
                func(*closure);
            };
        }
    };
}

}// namespace luisa::render

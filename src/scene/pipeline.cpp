//
// Created by Mike on 2021/12/15.
//

#include <scene/pipeline.h>

namespace luisa::render {

template<typename T, uint buffer_id_shift, uint buffer_element_alignment>
std::pair<BufferView<T>, uint> Pipeline::BufferArena<T, buffer_id_shift, buffer_element_alignment>::allocate(size_t n) noexcept {
    if (n > buffer_capacity) {// too big, will not use the arena
        auto buffer = _pipeline.create<Buffer<T>>(n);
        auto buffer_id = _pipeline.register_bindless(buffer->view());
        return std::make_pair(buffer->view(), buffer_id << buffer_id_shift);
    }
    static constexpr auto a = buffer_element_alignment;
    if (_buffer == nullptr || _buffer_offset + n > buffer_capacity) {
        _buffer = _pipeline.create<Buffer<T>>(buffer_capacity);
        _buffer_id = _pipeline.register_bindless(_buffer->view());
        _buffer_offset = 0u;
    }
    auto view = _buffer->view(_buffer_offset, n);
    auto id_and_offset = (_buffer_id << buffer_id_shift) |
                         (_buffer_offset / buffer_element_alignment);
    _buffer_offset = (_buffer_offset + n + a - 1u) / a * a;
    return std::make_pair(view, id_and_offset);
}

inline Pipeline::Pipeline(Device &device) noexcept
    : _device{device},
      _bindless_array{device.create_bindless_array(bindless_array_capacity)},
      _position_buffer_arena{*this},
      _attribute_buffer_arena{*this},
      _triangle_buffer_arena{*this},
      _area_cdf_buffer_arena{*this} {}

void Pipeline::_build_geometry(Stream &stream, luisa::span<const Shape *const> shapes, float init_time, AccelBuildHint hint) noexcept {
    _accel = _device.create_accel(hint);
    auto transform_builder = TransformTree::builder(init_time);
    for (auto shape : shapes) {
        _process_shape(stream, transform_builder, shape);
    }
    _transform_tree = transform_builder.build();
    _instance_buffer = _device.create_buffer<MeshInstance>(_instances.size());
    stream << _instance_buffer.copy_from(_instances.data())
           << _accel.build();
}

void Pipeline::_process_shape(
    Stream &stream, TransformTree::Builder &transform_builder, const Shape *shape,
    const Material *overridden_material, const Light *overridden_light) noexcept {

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
            auto [position_buffer_view, position_buffer_id_and_offset] = _position_buffer_arena.allocate(positions.size());
            auto [triangle_buffer_view, triangle_buffer_id_and_offset] = _triangle_buffer_arena.allocate(triangles.size());
            mesh.resource = create<Mesh>(position_buffer_view, triangle_buffer_view, shape->build_hint());
            stream << mesh.resource->build();
            // assign mesh data
            mesh.position_buffer_id_and_offset = position_buffer_id_and_offset;
            mesh.triangle_buffer_id_and_offset = triangle_buffer_id_and_offset;
            mesh.triangle_count = triangles.size();
            auto [_, attribute_buffer_id_and_offset] = _attribute_buffer_arena.allocate(attributes.size());
            mesh.attribute_buffer_id_and_offset = attribute_buffer_id_and_offset;
            // compute area cdf
            auto sum_area = 0.0;
            luisa::vector<float> areas;
            areas.reserve(triangles.size() + 1u);
            for (auto t : triangles) {
                auto p0 = positions[t.i0];
                auto p1 = positions[t.i1];
                auto p2 = positions[t.i2];
                auto v = cross(p1 - p0, p2 - p0);
                auto a = std::sqrt(dot(v, v));
                areas.emplace_back(static_cast<float>(sum_area));
                sum_area += a;
            }
            auto inv_sum_area = 1.0 / sum_area;
            for (auto &a : areas) { a = static_cast<float>(a * inv_sum_area); }
            areas.emplace_back(1.0f);
            auto [area_cdf_buffer_view, area_cdf_buffer_id_and_offset] = _area_cdf_buffer_arena.allocate(areas.size());
            mesh.area_cdf_buffer_id_and_offset = area_cdf_buffer_id_and_offset;
            stream << area_cdf_buffer_view.copy_from(areas.data());
            iter = _meshes.emplace(shape, mesh).first;
        }
        auto mesh = iter->second;

        // create instance
        MeshInstance instance{};
        instance.position_buffer_id_and_offset = mesh.position_buffer_id_and_offset;
        instance.attribute_buffer_id_and_offset = mesh.attribute_buffer_id_and_offset;
        instance.triangle_buffer_id_and_offset = mesh.triangle_buffer_id_and_offset;
        instance.triangle_buffer_size = mesh.triangle_count;
        instance.area_cdf_buffer_id_and_offset = mesh.area_cdf_buffer_id_and_offset;
        auto [m, m_flags] = _process_material(stream, material);
        auto [l, l_flags] = _process_light(stream, shape, light);
        if (m_flags > 0xffff'ffffu || l_flags > 0xffff'ffffu) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid material and/or light "
                "property flags: 0x{:08x} and 0x{:08x}.",
                m_flags, l_flags);
        }
        instance.material_and_light_property_flags = (l_flags << 16u) | m_flags;
        instance.material_buffer_id_and_tag = m;
        instance.light_buffer_id_and_tag = l;
        // add instance
        auto object_to_world = transform_builder.leaf(shape->transform(), _accel.size());
        _accel.emplace_back(*mesh.resource, object_to_world, true);
        _instances.emplace_back(instance);
    } else {
        if (shape->transform() != nullptr) { transform_builder.push(shape->transform()); }
        for (auto child : shape->children()) { _process_shape(stream, transform_builder, child, material, light); }
        if (shape->transform() != nullptr) { transform_builder.pop(); }
    }
}

std::pair<uint, uint> Pipeline::_process_material(Stream &stream, const Material *material) noexcept {
    if (material == nullptr) { return {~0u, Material::property_flag_black}; }
    if (auto iter = _materials.find(material); iter != _materials.cend()) {
        return iter->second;
    }
    auto tag = [this, material] {
        luisa::string impl_type{material->impl_type()};
        if (auto iter = _material_tags.find(impl_type);
            iter != _material_tags.cend()) {
            return iter->second;
        }
        static constexpr auto max_tag = (1u << MeshInstance::material_buffer_id_shift) - 1u;
        auto t = static_cast<uint32_t>(_material_interfaces.size());
        if (t > max_tag) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Too many materials."); }
        _material_interfaces.emplace_back(material->create_interface());
        _material_tags.emplace(std::move(impl_type), t);
        return t;
    }();
    auto buffer_id = material->encode(stream, *this);
    auto buffer_id_and_tag = (buffer_id << MeshInstance::material_buffer_id_shift) | tag;
    auto flags = material->property_flags();
    return _materials.emplace(material, std::make_pair(buffer_id_and_tag, flags)).first->second;
}

std::pair<uint, uint> Pipeline::_process_light(Stream &stream, const Shape *shape, const Light *light) noexcept {
    if (light == nullptr) { return std::make_pair(~0u, Light::property_flag_black); }
    return {};
}

luisa::unique_ptr<Pipeline> Pipeline::create(Device &device, Stream &stream, const Scene &scene) noexcept {
    auto pipeline = luisa::make_unique<Pipeline>(device);
    pipeline->_cameras.reserve(scene.cameras().size());
    pipeline->_films.reserve(scene.cameras().size());
    pipeline->_filters.reserve(scene.cameras().size());
    auto mean_time = 0.0;
    for (auto camera : scene.cameras()) {
        pipeline->_cameras.emplace_back(camera->build(stream, *pipeline));
        pipeline->_films.emplace_back(camera->film()->build(stream, *pipeline));
        pipeline->_filters.emplace_back(
            camera->filter() == nullptr ?
                nullptr :
                camera->filter()->build(stream, *pipeline));
        mean_time += (camera->time_span().x + camera->time_span().y) * 0.5f;
    }
    mean_time *= 1.0 / static_cast<double>(scene.cameras().size());
    pipeline->_build_geometry(stream, scene.shapes(), static_cast<float>(mean_time), AccelBuildHint::FAST_TRACE);
    pipeline->_integrator = scene.integrator()->build(stream, *pipeline);
    pipeline->_sampler = scene.integrator()->sampler()->build(stream, *pipeline);
    return pipeline;
}

void Pipeline::update(Stream &stream, float time) noexcept {
    // TODO: support deformable meshes
    _transform_tree.update(_accel, time);
    stream << _accel.update();
}

void Pipeline::render(Stream &stream) noexcept {
    _integrator->render(stream, *this);
}

Pipeline::~Pipeline() noexcept = default;

}// namespace luisa::render

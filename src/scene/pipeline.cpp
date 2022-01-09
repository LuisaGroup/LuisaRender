//
// Created by Mike on 2021/12/15.
//

#include <luisa-compute.h>
#include <scene/pipeline.h>

namespace luisa::render {

template<typename T, size_t capacity>
inline std::pair<BufferView<T>, uint> Pipeline::BufferArena<T, capacity>::allocate(size_t n) noexcept {
    if (n > capacity) {// too big, will not use the arena
        auto buffer = _pipeline.create<Buffer<T>>(n);
        auto buffer_id = _pipeline.register_bindless(buffer->view());
        return std::make_pair(buffer->view(), buffer_id);
    }
    if (_buffer == nullptr || _buffer_offset + n > capacity) {
        _buffer = _pipeline.create<Buffer<T>>(capacity);
        _buffer_id = _pipeline.register_bindless(_buffer->view());
        _buffer_offset = 0u;
    }
    auto view = _buffer->view(_buffer_offset, n);
    _buffer_offset += n;
    return std::make_pair(view, _buffer_id);
}

inline Pipeline::Pipeline(Device &device) noexcept
    : _device{device},
      _bindless_array{device.create_bindless_array(bindless_array_capacity)},
      _position_buffer_arena{*this},
      _attribute_buffer_arena{*this},
      _area_cdf_buffer_arena{*this} {}

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
            auto [position_buffer_view, position_buffer_id] = _position_buffer_arena.allocate(positions.size());
            auto [attribute_buffer_view, attribute_buffer_id] = _attribute_buffer_arena.allocate(attributes.size());
            if (position_buffer_view.offset() != attribute_buffer_view.offset()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION("Position and attribute buffer offsets mismatch.");
            }
            auto index_offset = static_cast<uint>(position_buffer_view.offset());
            luisa::vector<Triangle> offset_triangles(triangles.size());
            std::transform(triangles.cbegin(), triangles.cend(), offset_triangles.begin(), [index_offset](auto t) noexcept {
                return Triangle{t.i0 + index_offset, t.i1 + index_offset, t.i2 + index_offset};
            });
            auto triangle_buffer = create<Buffer<Triangle>>(triangles.size());
            auto triangle_buffer_id = register_bindless(triangle_buffer->view());
            mesh.resource = create<Mesh>(position_buffer_view.original(), *triangle_buffer, shape->build_hint());
            command_buffer << position_buffer_view.copy_from(positions.data())
                           << triangle_buffer->copy_from(offset_triangles.data())
                           << attribute_buffer_view.copy_from(attributes.data())
                           << mesh.resource->build();
            // assign mesh data
            mesh.position_buffer_id = position_buffer_id;
            mesh.triangle_buffer_id = triangle_buffer_id;
            mesh.triangle_count = triangles.size();
            mesh.attribute_buffer_id = attribute_buffer_id;
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
            auto [area_cdf_buffer_view, area_cdf_buffer_id] = _area_cdf_buffer_arena.allocate(areas.size());
            mesh.area_cdf_buffer_id_and_offset = (area_cdf_buffer_id << InstancedShape::area_cdf_buffer_id_shift) |
                                                 static_cast<uint>(area_cdf_buffer_view.offset() /
                                                                   InstancedShape::area_cdf_buffer_element_alignment);
            command_buffer << area_cdf_buffer_view.copy_from(areas.data())
                           << luisa::compute::commit();
            iter = _meshes.emplace(shape, mesh).first;
        }
        auto mesh = iter->second;

        // create instance
        InstancedShape instance{};
        instance.position_buffer_id = mesh.position_buffer_id;
        instance.attribute_buffer_id = mesh.attribute_buffer_id;
        instance.triangle_buffer_id = mesh.triangle_buffer_id;
        instance.triangle_count = mesh.triangle_count;
        instance.area_cdf_buffer_id_and_offset = mesh.area_cdf_buffer_id_and_offset;
        auto [m, m_flags] = _process_material(command_buffer, material);
        auto [l, l_flags] = _process_light(command_buffer, shape, light);
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
        for (auto child : shape->children()) { _process_shape(command_buffer, transform_builder, child, material, light); }
        if (shape->transform() != nullptr) { transform_builder.pop(); }
    }
}

std::pair<uint, uint> Pipeline::_process_material(CommandBuffer &command_buffer, const Material *material) noexcept {
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
        static constexpr auto max_tag = (1u << InstancedShape::material_buffer_id_shift) - 1u;
        auto t = static_cast<uint32_t>(_material_interfaces.size());
        if (t > max_tag) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Too many materials."); }
        _material_interfaces.emplace_back(material->interface());
        _material_tags.emplace(std::move(impl_type), t);
        return t;
    }();
    auto buffer_id = material->encode(*this, command_buffer);
    auto buffer_id_and_tag = (buffer_id << InstancedShape::material_buffer_id_shift) | tag;
    auto flags = material->property_flags();
    return _materials.emplace(material, std::make_pair(buffer_id_and_tag, flags)).first->second;
}

std::pair<uint, uint> Pipeline::_process_light(CommandBuffer &command_buffer, const Shape *shape, const Light *light) noexcept {
    // TODO...
    if (light == nullptr) { return std::make_pair(~0u, Light::property_flag_black); }
    return {};
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
            auto filter = camera->filter();
            pipeline->_filters.emplace_back(
                filter == nullptr ? nullptr : filter->build(*pipeline, command_buffer));
            mean_time += (camera->time_span().x + camera->time_span().y) * 0.5f;
        }
        mean_time *= 1.0 / static_cast<double>(scene.cameras().size());
        pipeline->_build_geometry(command_buffer, scene.shapes(), static_cast<float>(mean_time), AccelBuildHint::FAST_TRACE);
        pipeline->_integrator = scene.integrator()->build(*pipeline, command_buffer);
        pipeline->_sampler = scene.integrator()->sampler()->build(*pipeline, command_buffer);
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
    return buffer<Triangle>(instance->triangle_buffer_id).read(hit.prim);
}

std::pair<Var<float3>, Var<float3>> Pipeline::vertex(const Var<InstancedShape> &instance, const Var<float4x4> &shape_to_world, const Var<float3x3> &shape_to_world_normal, const Var<Triangle> &triangle, const Var<Hit> &hit) const noexcept {
    auto p0 = buffer<float3>(instance->position_buffer_id).read(triangle.i0);
    auto p1 = buffer<float3>(instance->position_buffer_id).read(triangle.i1);
    auto p2 = buffer<float3>(instance->position_buffer_id).read(triangle.i2);
    auto p = make_float3(shape_to_world * make_float4(hit->interpolate(p0, p1, p2), 1.0f));
    auto ng = normalize(shape_to_world_normal * cross(p1 - p0, p2 - p0));
    return std::make_pair(std::move(p), std::move(ng));
}

std::tuple<Var<float3>, Var<float3>, Var<float2>> Pipeline::vertex_attributes(
    const Var<InstancedShape> &instance, const Var<float3x3> &shape_to_world_normal, const Var<Triangle> &triangle, const Var<Hit> &hit) const noexcept {
    auto a0 = buffer<VertexAttribute>(instance->attribute_buffer_id).read(triangle.i0);
    auto a1 = buffer<VertexAttribute>(instance->attribute_buffer_id).read(triangle.i1);
    auto a2 = buffer<VertexAttribute>(instance->attribute_buffer_id).read(triangle.i2);
    auto normal = normalize(shape_to_world_normal * hit->interpolate(a0->normal(), a1->normal(), a2->normal()));
    auto tangent = normalize(shape_to_world_normal * hit->interpolate(a0->tangent(), a1->tangent(), a2->tangent()));
    auto uv = hit->interpolate(a0->uv(), a1->uv(), a2->uv());
    return std::make_tuple(std::move(normal), std::move(tangent), std::move(uv));
}

Var<Hit> Pipeline::trace_closest(const Var<Ray> &ray) const noexcept { return _accel.trace_closest(ray); }
Var<bool> Pipeline::trace_any(const Var<Ray> &ray) const noexcept { return _accel.trace_any(ray); }

Interaction Pipeline::interaction(const Var<Ray> &ray, const Var<Hit> &hit) const noexcept {
    using namespace luisa::compute;
    Interaction it;
    $if (!hit->miss()) {
        auto [shape, shape_to_world] = instance(hit);
        auto shape_to_world_normal = transpose(inverse(make_float3x3(shape_to_world)));
        auto tri = triangle(shape, hit);
        auto [p, ng] = vertex(shape, shape_to_world, shape_to_world_normal, tri, hit);
        auto [ns, tangent, uv] = vertex_attributes(shape, shape_to_world_normal, tri, hit);
        it = Interaction{shape, shape_to_world, p, -def<float3>(ray.direction), ng, uv, ns, tangent};
    };
    return it;
}

Pipeline::~Pipeline() noexcept = default;

}// namespace luisa::render

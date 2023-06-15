//
// Created by Mike on 2021/12/15.
//

#include <util/thread_pool.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

inline Pipeline::Pipeline(Device &device) noexcept
    : _device{device},
      _bindless_array{device.create_bindless_array(bindless_array_capacity)},
      _general_buffer_arena{luisa::make_unique<BufferArena>(device, 16_M)},
      _printer{luisa::make_unique<compute::Printer>(device)} {}

Pipeline::~Pipeline() noexcept = default;

uint Pipeline::register_surface(CommandBuffer &command_buffer, const Surface *surface) noexcept {
    if (auto iter = _surface_tags.find(surface);
        iter != _surface_tags.end()) { return iter->second; }
    auto tag = _surfaces.emplace(surface->build(*this, command_buffer));
    _surface_tags.emplace(surface, tag);
    return tag;
}

uint Pipeline::register_light(CommandBuffer &command_buffer, const Light *light) noexcept {
    if (auto iter = _light_tags.find(light);
        iter != _light_tags.end()) { return iter->second; }
    auto tag = _lights.emplace(light->build(*this, command_buffer));
    _light_tags.emplace(light, tag);
    return tag;
}

uint Pipeline::register_medium(CommandBuffer &command_buffer, const Medium *medium) noexcept {
    if (auto iter = _medium_tags.find(medium);
        iter != _medium_tags.end()) { return iter->second; }
    auto tag = _media.emplace(medium->build(*this, command_buffer));
    _medium_tags.emplace(medium, tag);
    return tag;
}

luisa::unique_ptr<Pipeline> Pipeline::create(Device &device, Stream &stream, const Scene &scene) noexcept {
    global_thread_pool().synchronize();
    auto pipeline = luisa::make_unique<Pipeline>(device);
    pipeline->_transform_matrices.resize(transform_matrix_buffer_size);
    pipeline->_transform_matrix_buffer = device.create_buffer<float4x4>(transform_matrix_buffer_size);
    stream << pipeline->printer().reset();
    auto initial_time = std::numeric_limits<float>::max();
    for (auto c : scene.cameras()) {
        if (c->shutter_span().x < initial_time) {
            initial_time = c->shutter_span().x;
        }
    }
    pipeline->_initial_time = initial_time;
    pipeline->_cameras.reserve(scene.cameras().size());

    CommandBuffer command_buffer{&stream};
    auto update_bindless_if_dirty = [&pipeline, &command_buffer] {
        if (pipeline->_bindless_array.dirty()) {
            command_buffer << pipeline->_bindless_array.update();
        }
    };
    pipeline->_spectrum = scene.spectrum()->build(*pipeline, command_buffer);
    update_bindless_if_dirty();
    for (auto camera : scene.cameras()) {
        pipeline->_cameras.emplace_back(camera->build(*pipeline, command_buffer));
    }
    update_bindless_if_dirty();
    pipeline->_geometry = luisa::make_unique<Geometry>(*pipeline);
    pipeline->_geometry->build(command_buffer, scene.shapes(), pipeline->_initial_time);
    update_bindless_if_dirty();
    if (auto env = scene.environment(); env != nullptr && !env->is_black()) {
        pipeline->_environment = env->build(*pipeline, command_buffer);
    }
    if (auto environment_medium = scene.environment_medium(); environment_medium != nullptr) {
        pipeline->_environment_medium_tag = pipeline->register_medium(command_buffer, environment_medium);
    }
    if (pipeline->_lights.empty() && pipeline->_environment == nullptr) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights or environment found in the scene.");
    }
    update_bindless_if_dirty();
    pipeline->_integrator = scene.integrator()->build(*pipeline, command_buffer);
    if (!pipeline->_transforms.empty()) {
        command_buffer << pipeline->_transform_matrix_buffer.view(0u, pipeline->_transforms.size())
                              .copy_from(pipeline->_transform_matrices.data());
    }
    update_bindless_if_dirty();
    command_buffer << compute::commit();
    LUISA_INFO("Created pipeline with {} camera(s), {} shape instance(s), "
               "{} surface instance(s), and {} light instance(s).",
               pipeline->_cameras.size(),
               pipeline->_geometry->instances().size(),
               pipeline->_surfaces.size(),
               pipeline->_lights.size());
    return pipeline;
}

bool Pipeline::update(CommandBuffer &command_buffer, float time) noexcept {
    // TODO: support deformable meshes
    auto updated = _geometry->update(command_buffer, time);
    if (_any_dynamic_transforms) {
        updated = true;
        for (auto i = 0u; i < _transforms.size(); ++i) {
            _transform_matrices[i] = _transforms[i]->matrix(time);
        }
        command_buffer << _transform_matrix_buffer.view(0u, _transforms.size())
                              .copy_from(_transform_matrices.data());
    }
    return updated;
}

void Pipeline::render(Stream &stream) noexcept {
    _integrator->render(stream);
}

const Texture::Instance *Pipeline::build_texture(CommandBuffer &command_buffer, const Texture *texture) noexcept {
    if (texture == nullptr) { return nullptr; }
    if (auto iter = _textures.find(texture); iter != _textures.end()) {
        return iter->second.get();
    }
    auto t = texture->build(*this, command_buffer);
    return _textures.emplace(texture, std::move(t)).first->second.get();
}

const Filter::Instance *Pipeline::build_filter(CommandBuffer &command_buffer, const Filter *filter) noexcept {
    if (filter == nullptr) { return nullptr; }
    if (auto iter = _filters.find(filter); iter != _filters.end()) {
        return iter->second.get();
    }
    auto f = filter->build(*this, command_buffer);
    return _filters.emplace(filter, std::move(f)).first->second.get();
}

const PhaseFunction::Instance *Pipeline::build_phasefunction(CommandBuffer &command_buffer, const PhaseFunction *phasefunction) noexcept {
    if (phasefunction == nullptr) { return nullptr; }
    if (auto iter = _phasefunctions.find(phasefunction); iter != _phasefunctions.end()) {
        return iter->second.get();
    }
    auto pf = phasefunction->build(*this, command_buffer);
    return _phasefunctions.emplace(phasefunction, std::move(pf)).first->second.get();
}

void Pipeline::register_transform(const Transform *transform) noexcept {
    if (transform == nullptr) { return; }
    if (!_transform_to_id.contains(transform)) {
        auto transform_id = static_cast<uint>(_transforms.size());
        LUISA_ASSERT(transform_id < transform_matrix_buffer_size,
                     "Transform matrix buffer overflows.");
        _transform_to_id.emplace(transform, transform_id);
        _transforms.emplace_back(transform);
        _any_dynamic_transforms |= !transform->is_static();
        _transform_matrices[transform_id] = transform->matrix(_initial_time);
    }
}

Float4x4 Pipeline::transform(const Transform *transform) const noexcept {
    if (transform == nullptr) { return make_float4x4(1.f); }
    if (transform->is_identity()) { return make_float4x4(1.f); }
    auto iter = _transform_to_id.find(transform);
    LUISA_ASSERT(iter != _transform_to_id.cend(), "Transform is not registered.");
    return _transform_matrix_buffer->read(iter->second);
}

uint Pipeline::named_id(luisa::string_view name) const noexcept {
    auto iter = _named_ids.find(name);
    LUISA_ASSERT(iter != _named_ids.cend(),
                 "Named ID '{}' not found.", name);
    return iter->second;
}

std::pair<BufferView<float4>, uint> Pipeline::allocate_constant_slot() noexcept {
    if (!_constant_buffer) {
        _constant_buffer = device().create_buffer<float4>(constant_buffer_size);
    }
    auto slot = _constant_count++;
    LUISA_ASSERT(slot < constant_buffer_size,
                 "Constant buffer overflows.");
    return {_constant_buffer.view(static_cast<uint>(slot), 1u),
            static_cast<uint>(slot)};
}

Float4 Pipeline::constant(Expr<uint> index) const noexcept {
    return _constant_buffer->read(index);
}

}// namespace luisa::render

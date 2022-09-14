//
// Created by Mike on 2021/12/15.
//

#include <luisa-compute.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

inline Pipeline::Pipeline(Device &device) noexcept
    : _device{device},
      _bindless_array{device.create_bindless_array(bindless_array_capacity)},
      _general_buffer_arena{luisa::make_unique<BufferArena>(device, 16_mb)},
      _printer{compute::Printer{device}} {}

Pipeline::~Pipeline() noexcept = default;

uint Pipeline::register_surface(CommandBuffer &command_buffer, const Surface *surface) noexcept {
    auto [iter, not_existent] = _surface_tags.try_emplace(surface, 0u);
    if (not_existent) { iter->second = _surfaces.emplace(surface->build(*this, command_buffer)); }
    return iter->second;
}

uint Pipeline::register_light(CommandBuffer &command_buffer, const Light *light) noexcept {
    auto [iter, not_existent] = _light_tags.try_emplace(light, 0u);
    if (not_existent) { iter->second = _lights.emplace(light->build(*this, command_buffer)); }
    return iter->second;
}

luisa::unique_ptr<Pipeline> Pipeline::create(Device &device, Stream &stream, const Scene &scene) noexcept {
    ThreadPool::global().synchronize();
    auto pipeline = luisa::make_unique<Pipeline>(device);
    auto initial_time = std::numeric_limits<float>::max();
    for (auto c : scene.cameras()) {
        if (c->shutter_span().x < initial_time) {
            initial_time = c->shutter_span().x;
        }
    }
    pipeline->_initial_time = initial_time;
    pipeline->_transform_matrices.resize(transform_matrix_buffer_size);
    pipeline->_transform_matrix_buffer = device.create_buffer<float4x4>(transform_matrix_buffer_size);
    if (scene.integrator()->is_differentiable()) {
        pipeline->_differentiation =
            luisa::make_unique<Differentiation>(*pipeline);
    }
    pipeline->_cameras.reserve(scene.cameras().size());
    auto command_buffer = stream.command_buffer();
    pipeline->_spectrum = scene.spectrum()->build(*pipeline, command_buffer);
    for (auto camera : scene.cameras()) {
        pipeline->_cameras.emplace_back(camera->build(*pipeline, command_buffer));
    }
    pipeline->_geometry = luisa::make_unique<Geometry>(*pipeline);
    pipeline->_geometry->build(command_buffer, scene.shapes(), pipeline->_initial_time, AccelUsageHint::FAST_TRACE);
    if (auto env = scene.environment(); env != nullptr && !env->is_black()) {
        pipeline->_environment = env->build(*pipeline, command_buffer);
    }
    if (pipeline->_lights.empty() && pipeline->_environment == nullptr) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights or environment found in the scene.");
    }
    pipeline->_integrator = scene.integrator()->build(*pipeline, command_buffer);
    command_buffer << pipeline->_bindless_array.update();
    if (auto &&diff = pipeline->_differentiation) {
        diff->register_optimizer(dynamic_cast<DifferentiableIntegrator::Instance *>(pipeline->_integrator.get())->optimizer());
        diff->materialize(command_buffer);
    }
    if (!pipeline->_transforms.empty()) {
        command_buffer << pipeline->_transform_matrix_buffer.view(0u, pipeline->_transforms.size())
                              .copy_from(pipeline->_transform_matrices.data());
    }
    command_buffer << commit();
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
    auto [iter, not_exists] = _textures.try_emplace(texture, nullptr);
    if (not_exists) { iter->second = texture->build(*this, command_buffer); }
    return iter->second.get();
}

const Filter::Instance *Pipeline::build_filter(CommandBuffer &command_buffer, const Filter *filter) noexcept {
    if (filter == nullptr) { return nullptr; }
    auto [iter, not_exists] = _filters.try_emplace(filter, nullptr);
    if (not_exists) { iter->second = filter->build(*this, command_buffer); }
    return iter->second.get();
}

Differentiation *Pipeline::differentiation() noexcept {
    LUISA_ASSERT(_differentiation, "Differentiation is not constructed.");
    return _differentiation.get();
}

const Differentiation *Pipeline::differentiation() const noexcept {
    LUISA_ASSERT(_differentiation, "Differentiation is not constructed.");
    return _differentiation.get();
}

void Pipeline::register_transform(const Transform *transform) noexcept {
    if (transform == nullptr) { return; }
    auto [iter, success] = _transform_to_id.try_emplace(
        transform, static_cast<uint>(_transforms.size()));
    LUISA_ASSERT(iter->second < transform_matrix_buffer_size,
                 "Transform matrix buffer overflows.");
    if (success) [[likely]] {
        _transforms.push_back(transform);
        _any_dynamic_transforms |= !transform->is_static();
        _transform_matrices[iter->second] = transform->matrix(_initial_time);
    }
}

Float4x4 Pipeline::transform(const Transform *transform) const noexcept {
    if (transform == nullptr) { return make_float4x4(1.f); }
    auto iter = _transform_to_id.find(transform);
    LUISA_ASSERT(iter != _transform_to_id.cend(), "Transform is not registered.");
    return _transform_matrix_buffer.read(iter->second);
}

uint Pipeline::named_id(luisa::string_view name) const noexcept {
    auto iter = _named_ids.find(name);
    LUISA_ASSERT(iter != _named_ids.cend(),
                 "Named ID '{}' not found.", name);
    return iter->second;
}

}// namespace luisa::render

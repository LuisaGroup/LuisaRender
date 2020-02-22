//
// Created by Mike Smith on 2020/2/21.
//

#include "illumination.h"
#include "sampler.h"

namespace luisa {

Illumination::Illumination(Device *device, std::vector<std::shared_ptr<Light>> lights, Geometry *geometry)
    : _device{device},
      _geometry{geometry},
      _lights{std::move(lights)},
      _info_buffer{device->create_buffer<illumination::Info>(lights.size(), BufferStorage::MANAGED)},
      _uniform_select_lights_kernel{device->create_kernel("illumination_uniform_select_lights")} {
    
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> light_counts{};
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> light_data_strides{};
    for (auto i = 0ul; i < lights.size(); i++) {
        auto &&light = lights[i];
        auto tag = light->tag();
        if (light_counts[tag] == 0u) {
            LUISA_ERROR_IF_NOT(tag == _light_sampling_kernels.size(), "incorrect light tag assigned: ", tag);
            _light_sampling_kernels.emplace_back(light->create_generate_samples_kernel());
            _light_sampling_dispatches.emplace_back(light->create_generate_samples_dispatch());
            _light_sampling_dimensions.emplace_back(light->sampling_dimensions());
            light_data_strides[tag] = light->data_stride();
        }
        _info_buffer->view()[i] = {tag, light_counts[tag]++};
    }
    _info_buffer->upload();
    
    // encode per-light data
    std::array<uint, Light::MAX_LIGHT_TAG_COUNT> encoded_light_counts{};
    _light_data_buffers.reserve(_light_sampling_kernels.size());
    for (auto tag = 0u; tag < _light_sampling_kernels.size(); tag++) {
        _light_data_buffers.emplace_back(_device->allocate_buffer(light_data_strides[tag] * light_counts[tag], BufferStorage::MANAGED));
    }
    for (auto &&light : lights) { light->encode_data(*_light_data_buffers[light->tag()], encoded_light_counts[light->tag()]++); }
    for (auto &&buffer : _light_data_buffers) { buffer->upload(); }
}

void Illumination::uniform_select_lights(KernelDispatcher &dispatch,
                                         uint max_ray_count,
                                         BufferView<uint> ray_queue,
                                         BufferView<uint> ray_queue_size,
                                         Sampler &sampler,
                                         BufferView<Selection> queues,
                                         BufferView<uint> queue_sizes) {
    
    LUISA_ERROR_IF(queue_sizes.size() < tag_count(), "no enough space in queue_sizes");
    LUISA_ERROR_IF(queues.size() < tag_count() * max_ray_count, "no enough space in queues");
    
    auto sample_buffer = sampler.generate_samples(dispatch, 1u, ray_queue, ray_queue_size);
    dispatch(*_uniform_select_lights_kernel, max_ray_count, [&](KernelArgumentEncoder &encode) {
        encode("sample_buffer", sample_buffer);
        encode("info_buffer", _info_buffer->view());
        encode("queue_sizes", queue_sizes);
        encode("queues", queues);
        encode("ray_count", ray_queue_size);
        encode("uniforms", illumination::SelectLightsKernelUniforms{static_cast<uint>(_lights.size()), max_ray_count});
    });
}

void Illumination::sample_lights(KernelDispatcher &dispatch,
                                 Sampler &sampler,
                                 BufferView<uint> ray_indices,
                                 BufferView<uint> ray_count,
                                 BufferView<Selection> queues,
                                 BufferView<uint> queue_sizes,
                                 uint max_queue_size,
                                 InteractionBufferSet &interactions,
                                 LightSampleBufferSet &light_samples) {
    
    for (auto tag = 0ul; tag < tag_count(); tag++) {
        auto queue = queues.subview(tag * max_queue_size, max_queue_size);
        auto queue_size = queue_sizes.subview(tag, 1u);
        auto &&light_data = *_light_data_buffers[tag];
        auto &&kernel = *_light_sampling_kernels[tag];
        auto samples = sampler.generate_samples(dispatch, _light_sampling_dimensions[tag], ray_indices, ray_count);
        _light_sampling_dispatches[tag](dispatch, kernel, max_queue_size, samples, light_data, queue, queue_size, interactions, _geometry, light_samples);
    }
    
}

}

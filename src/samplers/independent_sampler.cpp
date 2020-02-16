//
// Created by Mike Smith on 2020/2/14.
//

#include "independent_sampler.h"

namespace luisa {

IndependentSampler::IndependentSampler(Device *device, const ParameterSet &parameter_set)
    : Sampler{device, parameter_set},
      _reset_states_kernel{device->create_kernel("independent_sampler_prepare")},
      _generate_1d_samples_kernel{device->create_kernel("independent_sampler_generate_1d_samples")},
      _generate_2d_samples_kernel{device->create_kernel("independent_sampler_generate_2d_samples")},
      _generate_3d_samples_kernel{device->create_kernel("independent_sampler_generate_3d_samples")},
      _generate_4d_samples_kernel{device->create_kernel("independent_sampler_generate_4d_samples")} {}

void IndependentSampler::generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float> sample_buffer) {
    dispatch(*_generate_1d_samples_kernel, ray_queue_buffer.size(), [&](KernelArgumentEncoder &encode) {
        encode("sampler_state_buffer", *_state_buffer);
        encode("ray_queue", ray_queue_buffer);
        encode("ray_count", ray_count_buffer);
        encode("sample_buffer", sample_buffer);
    });
    _current_dimension++;
}

void IndependentSampler::generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float2> sample_buffer) {
    dispatch(*_generate_2d_samples_kernel, ray_queue_buffer.size(), [&](KernelArgumentEncoder &encode) {
        encode("sampler_state_buffer", *_state_buffer);
        encode("ray_queue", ray_queue_buffer);
        encode("ray_count", ray_count_buffer);
        encode("sample_buffer", sample_buffer);
    });
    _current_dimension += 2u;
}

void IndependentSampler::generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float3> sample_buffer) {
    dispatch(*_generate_3d_samples_kernel, ray_queue_buffer.size(), [&](KernelArgumentEncoder &encode) {
        encode("sampler_state_buffer", *_state_buffer);
        encode("ray_queue", ray_queue_buffer);
        encode("ray_count", ray_count_buffer);
        encode("sample_buffer", sample_buffer);
    });
    _current_dimension += 3u;
}

void IndependentSampler::generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float4> sample_buffer) {
    dispatch(*_generate_4d_samples_kernel, ray_queue_buffer.size(), [&](KernelArgumentEncoder &encode) {
        encode("sampler_state_buffer", *_state_buffer);
        encode("ray_queue", ray_queue_buffer);
        encode("ray_count", ray_count_buffer);
        encode("sample_buffer", sample_buffer);
    });
    _current_dimension += 4u;
}

void IndependentSampler::reset_states(KernelDispatcher &dispatch, uint2 film_resolution) {
    auto pixel_count = film_resolution.x * film_resolution.y;
    Sampler::reset_states(dispatch, film_resolution);
    if (_state_buffer == nullptr || _state_buffer->view().size() < pixel_count) {
        _state_buffer = _device->create_buffer<independent_sampler::SamplerState>(pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    dispatch(*_reset_states_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
        encode("film_resolution", _film_resolution);
        encode("sampler_state_buffer", *_state_buffer);
    });
}

}

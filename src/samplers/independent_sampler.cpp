//
// Created by Mike Smith on 2020/2/14.
//

#include "independent_sampler.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("Independent", IndependentSampler)

IndependentSampler::IndependentSampler(Device *device, const ParameterSet &parameter_set)
    : Sampler{device, parameter_set},
      _reset_states_kernel{device->create_kernel("independent_sampler_reset_states")},
      _generate_samples_kernel{device->create_kernel("independent_sampler_generate_samples")} {}

void IndependentSampler::_generate_samples(
    KernelDispatcher &dispatch,
    BufferView<uint> ray_queue_buffer,
    BufferView<uint> ray_count_buffer,
    BufferView<float> sample_buffer, uint d) {
    
    dispatch(*_generate_samples_kernel, ray_queue_buffer.size(), [&](KernelArgumentEncoder &encode) {
        encode("sampler_state_buffer", *_state_buffer);
        encode("ray_queue", ray_queue_buffer);
        encode("ray_count", ray_count_buffer);
        encode("sample_buffer", sample_buffer);
        encode("uniforms", sampler::independent::GenerateSamplesKernelUniforms{_tile_viewport, _film_viewport, d, true});
    });
}

void IndependentSampler::_reset_states() {
    auto size = _film_viewport.size.x * _film_viewport.size.y;
    if (_state_buffer == nullptr || _state_buffer->size() < size) {
        _state_buffer = _device->create_buffer<sampler::independent::State>(size, BufferStorage::DEVICE_PRIVATE);
    }
    _device->launch_async([&](KernelDispatcher &dispatch) {
        dispatch(*_reset_states_kernel, size, [&](KernelArgumentEncoder &encode) {
            encode("film_viewport", _film_viewport);
            encode("sampler_state_buffer", *_state_buffer);
        });
    });
}

void IndependentSampler::_generate_samples(KernelDispatcher &dispatch, BufferView<float> sample_buffer, uint d) {
    dispatch(*_generate_samples_kernel, _tile_viewport.size.x * _tile_viewport.size.y, [&](KernelArgumentEncoder &encode) {
        encode("sampler_state_buffer", *_state_buffer);
        encode("sample_buffer", sample_buffer);
        encode("uniforms", sampler::independent::GenerateSamplesKernelUniforms{_tile_viewport, _film_viewport, d, false});
    });
}

}

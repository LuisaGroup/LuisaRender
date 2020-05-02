//
// Created by Mike Smith on 2020/5/1.
//

#include "sampler.h"

namespace luisa {

BufferView<float> Sampler::generate_samples(KernelDispatcher &dispatch, uint dimensions) {
    LUISA_EXCEPTION_IF_NOT(dimensions >= 1 && dimensions <= 4, "Bad sample dimensions: ", dimensions);
    _generate_samples(dispatch, _sample_buffer->view_as<float>(), dimensions);
    return _sample_buffer->view_as<float>();
}

BufferView<float> Sampler::generate_samples(KernelDispatcher &dispatch, uint dimensions, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer) {
    LUISA_EXCEPTION_IF_NOT(dimensions >= 1 && dimensions <= 4, "Bad sample dimensions: ", dimensions);
    _generate_samples(dispatch, ray_queue_buffer, ray_count_buffer, _sample_buffer->view_as<float>(), dimensions);
    return _sample_buffer->view_as<float>();
}

void Sampler::prepare_for_tile(KernelDispatcher &dispatch, Viewport tile_viewport) {
    _tile_viewport = tile_viewport;
    _prepare_for_tile(dispatch);
}

void Sampler::start_next_frame(KernelDispatcher &dispatch) {
    _frame_index++;
    _start_next_frame(dispatch);
}

void Sampler::reset_states(uint2 film_resolution, Viewport film_viewport) {
    _frame_index = 0u;
    _film_resolution = film_resolution;
    _film_viewport = film_viewport;
    auto film_viewport_pixel_count = _film_viewport.size.x * _film_viewport.size.y;
    if (_sample_buffer == nullptr || _sample_buffer->size() < film_viewport_pixel_count) {
        _sample_buffer = _device->create_buffer<float4>(film_viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
    }
    _reset_states();
}

}
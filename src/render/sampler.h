//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#include "viewport.h"
#include "core/plugin.h"
#include "core/parser.h"
#include <compute/device.h>

namespace luisa {

class Sampler : public Plugin {

protected:
    uint _spp;
    uint _frame_index{};
    uint2 _film_resolution{};
    Viewport _film_viewport{};
    Viewport _tile_viewport{};
    std::unique_ptr<Buffer<float4>> _sample_buffer;
    
    virtual void _generate_samples(KernelDispatcher &dispatch, BufferView<float> sample_buffer, uint d) = 0;
    virtual void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float> sample_buffer, uint d) = 0;
    
    virtual void _reset_states() = 0;
    virtual void _start_next_frame(KernelDispatcher &dispatch) = 0;
    virtual void _prepare_for_tile(KernelDispatcher &dispatch) = 0;

public:
    Sampler(Device *device, const ParameterSet &parameter_set)
        : Plugin{device}, _spp{parameter_set["spp"].parse_uint_or_default(1024u)} {}
    
    [[nodiscard]] uint spp() const noexcept { return _spp; }
    [[nodiscard]] uint frame_index() const noexcept { return _frame_index; }
    
    void reset_states(uint2 film_resolution, Viewport film_viewport);
    void start_next_frame(KernelDispatcher &dispatch);
    void prepare_for_tile(KernelDispatcher &dispatch, Viewport tile_viewport);
    
    [[nodiscard]] BufferView<float> generate_samples(KernelDispatcher &dispatch, uint dimensions);
    [[nodiscard]] BufferView<float> generate_samples(KernelDispatcher &dispatch, uint dimensions, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer);
};

}

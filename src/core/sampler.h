//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#include "node.h"
#include "parser.h"
#include "device.h"

namespace luisa {

class Sampler : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Sampler);

protected:
    uint _spp;
    uint _current_dimension{};
    uint _frame_index{};
    uint2 _film_resolution{};

public:
    Sampler(Device *device, const ParameterSet &parameter_set)
        : Node{device}, _spp{parameter_set["spp"].parse_uint_or_default(1024u)} {}
    
    virtual void reset_states(KernelDispatcher &dispatch[[maybe_unused]], uint2 film_resolution) {
        _current_dimension = 0u;
        _frame_index = 0u;
        _film_resolution = film_resolution;
    }
    
    virtual void start_frame(KernelDispatcher &dispatch[[maybe_unused]], BufferView<uint> ray_queue_buffer[[maybe_unused]], BufferView<uint> ray_count_buffer[[maybe_unused]]) {
        _frame_index++;
        _current_dimension = 0u;
    }
    
    virtual void generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float> sample_buffer) = 0;
    virtual void generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float2> sample_buffer) = 0;
    virtual void generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float3> sample_buffer) = 0;
    virtual void generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float4> sample_buffer) = 0;
};

}

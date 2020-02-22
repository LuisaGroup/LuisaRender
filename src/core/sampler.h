//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#include "viewport.h"
#include "node.h"
#include "parser.h"
#include "device.h"

namespace luisa {

class Sampler : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Sampler);

protected:
    uint _spp;
    uint _frame_index{};
    uint2 _film_resolution{};
    Viewport _film_viewport{};
    Viewport _tile_viewport{};
    std::unique_ptr<Buffer<float4>> _sample_buffer;
    
    virtual void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float> sample_buffer) = 0;
    virtual void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float2> sample_buffer) = 0;
    virtual void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float3> sample_buffer) = 0;
    virtual void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float4> sample_buffer) = 0;
    
    virtual void _reset_states() = 0;
    virtual void _start_next_frame(KernelDispatcher &dispatch) = 0;
    virtual void _prepare_for_tile(KernelDispatcher &dispatch) = 0;

public:
    Sampler(Device *device, const ParameterSet &parameter_set)
        : Node{device}, _spp{parameter_set["spp"].parse_uint_or_default(1024u)} {}
    
    [[nodiscard]] uint spp() const noexcept { return _spp; }
    [[nodiscard]] uint frame_index() const noexcept { return _frame_index; }
    
    void reset_states(uint2 film_resolution, Viewport film_viewport) {
        _frame_index = 0u;
        _film_resolution = film_resolution;
        _film_viewport = film_viewport;
        auto film_viewport_pixel_count = _film_viewport.size.x * _film_viewport.size.y;
        if (_sample_buffer == nullptr || _sample_buffer->size() < film_viewport_pixel_count) {
            _sample_buffer = _device->create_buffer<float4>(film_viewport_pixel_count, BufferStorage::DEVICE_PRIVATE);
        }
        _reset_states();
    }
    
    void start_next_frame(KernelDispatcher &dispatch) {
        _frame_index++;
        _start_next_frame(dispatch);
    }
    
    void prepare_for_tile(KernelDispatcher &dispatch, Viewport tile_viewport) {
        _tile_viewport = tile_viewport;
        _prepare_for_tile(dispatch);
    }
    
    BufferView<float> generate_samples(KernelDispatcher &dispatch, uint dimensions, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer) {
        switch (dimensions) {
            case 0:
                break;
            case 1:
                _generate_samples(dispatch, ray_queue_buffer, ray_count_buffer, _sample_buffer->view_as<float>());
                break;
            case 2:
                _generate_samples(dispatch, ray_queue_buffer, ray_count_buffer, _sample_buffer->view_as<float2>());
                break;
            case 3:
                _generate_samples(dispatch, ray_queue_buffer, ray_count_buffer, _sample_buffer->view_as<float3>());
                break;
            case 4:
                _generate_samples(dispatch, ray_queue_buffer, ray_count_buffer, _sample_buffer->view_as<float4>());
                break;
            default:
                LUISA_ERROR("bad sample dimensions: ", dimensions);
        }
        return _sample_buffer->view_as<float>();
    }
    virtual BufferView<float4> generate_camera_samples(KernelDispatcher &dispatch) = 0;
};

}

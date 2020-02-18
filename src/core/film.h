//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "data_types.h"
#include "ray.h"

namespace luisa::film {

LUISA_DEVICE_CALLABLE inline void reset_accumulation_buffer(
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    uint pixel_count,
    uint tid) noexcept {
    
    if (tid < pixel_count) { accumulation_buffer[tid] = make_float4(); }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "device.h"
#include "node.h"
#include "filter.h"
#include "parser.h"
#include "viewport.h"

namespace luisa {

class Film : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Film);
    
protected:
    Viewport _film_viewport{};
    uint2 _resolution;
    std::shared_ptr<Filter> _filter;
    std::unique_ptr<Buffer<float4>> _accumulation_buffer;
    std::unique_ptr<Kernel> _reset_accumulation_buffer_kernel;

public:
    Film(Device *device, const ParameterSet &parameters)
        : Node{device},
          _resolution{parameters["resolution"].parse_uint2_or_default(make_uint2(1280, 720))},
          _filter{parameters["filter"].parse<Filter>()} {
        
        _accumulation_buffer = device->create_buffer<float4>(_resolution.x * _resolution.y, BufferStorage::DEVICE_PRIVATE);
        _reset_accumulation_buffer_kernel = device->create_kernel("film_reset_accumulation_buffer");
    }
    
    virtual void reset_accumulation_buffer(Viewport film_viewport) {
        _film_viewport = film_viewport;
        _device->launch_async([&](KernelDispatcher &dispatch) {
            auto pixel_count = _resolution.x * _resolution.y;
            dispatch(*_reset_accumulation_buffer_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
                encode("accumulation_buffer", *_accumulation_buffer);
                encode("pixel_count", pixel_count);
            });
        });
    }
    
    virtual void accumulate_tile(KernelDispatcher &dispatch, BufferView<float2> pixel_buffer, BufferView<float3> radiance_buffer, Viewport tile_viewport) {
        _filter->apply_and_accumulate(dispatch, _resolution, tile_viewport, tile_viewport, pixel_buffer, radiance_buffer, _accumulation_buffer->view());
    }
    
    virtual void postprocess(KernelDispatcher &dispatch) = 0;
    virtual void save(const std::filesystem::path &filename) = 0;
    [[nodiscard]] Filter &filter() noexcept { return *_filter; }
    [[nodiscard]] BufferView<float4> accumulation_buffer() noexcept { return _accumulation_buffer->view(); }
    [[nodiscard]] uint2 resolution() noexcept { return _resolution; }

};

}

#endif
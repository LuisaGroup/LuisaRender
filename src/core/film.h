//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "data_types.h"
#include "ray.h"

namespace luisa::film {

LUISA_DEVICE_CALLABLE inline void clear_accumulation_buffer(
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    uint pixel_count,
    uint tid) noexcept {
    
    if (tid < pixel_count) { accumulation_buffer[tid] = make_float4(); }
}

LUISA_DEVICE_CALLABLE inline void accumulate_frame(
    LUISA_DEVICE_SPACE const float4 *frame,
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    uint pixel_count,
    uint tid) noexcept {
    
    if (tid < pixel_count) { accumulation_buffer[tid] += frame[tid]; }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "device.h"
#include "node.h"
#include "filter.h"
#include "parser.h"

namespace luisa {

class Film : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Film);

protected:
    uint2 _resolution;
    std::shared_ptr<Filter> _filter;
    std::unique_ptr<Buffer<float4>> _accumulation_buffer;
    std::unique_ptr<Buffer<float4>> _framebuffer;
    std::unique_ptr<Kernel> _clear_accumulation_buffer_kernel;
    std::unique_ptr<Kernel> _accumulate_frame_kernel;

public:
    Film(Device *device, const ParameterSet &parameters)
        : Node{device},
          _resolution{parameters["resolution"].parse_uint2_or_default(make_uint2(1280, 720))},
          _filter{parameters["filter"].parse<Filter>()} {
        
        _accumulation_buffer = device->create_buffer<float4>(_resolution.x * _resolution.y, BufferStorage::DEVICE_PRIVATE);
        _framebuffer = device->create_buffer<float4>(_resolution.x * _resolution.y, BufferStorage::DEVICE_PRIVATE);
        _clear_accumulation_buffer_kernel = device->create_kernel("film_clear_accumulation_buffer");
        _accumulate_frame_kernel = device->create_kernel("film_accumulate_frame");
    }
    
    virtual void clear_accumulation_buffer(KernelDispatcher &dispatch) {
        auto pixel_count = _resolution.x * _resolution.y;
        dispatch(*_clear_accumulation_buffer_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
            encode("accumulation_buffer", *_accumulation_buffer);
            encode("pixel_count", pixel_count);
        });
    }
    
    virtual void accumulate_frame(KernelDispatcher &dispatch, BufferView<float2> pixel_buffer, BufferView<float3> radiance_buffer) {
        _filter->apply(dispatch, pixel_buffer, radiance_buffer, _framebuffer->view(), _resolution);
        auto pixel_count = _resolution.x * _resolution.y;
        dispatch(*_accumulate_frame_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
            encode("accumulation_buffer", *_accumulation_buffer);
            encode("frame", *_framebuffer);
            encode("pixel_count", pixel_count);
        });
    }
    
    virtual void postprocess(KernelDispatcher &dispatch) = 0;
    virtual void save(const std::filesystem::path &filename) = 0;
    [[nodiscard]] Filter &filter() noexcept { return *_filter; }
    [[nodiscard]] BufferView<float4> accumulation_buffer() noexcept { return _accumulation_buffer->view(); }
    [[nodiscard]] uint2 resolution() noexcept { return _resolution; }

};

}

#endif
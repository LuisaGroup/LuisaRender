//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include <compute/data_types.h>
#include "ray.h"
#include "viewport.h"

namespace luisa::film {

struct AccumulateTileKernelUniforms {
    Viewport tile_viewport;
    uint2 film_resolution;
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <compute/device.h>
#include "core/plugin.h"
#include "filter.h"
#include "core/parser.h"
#include "viewport.h"

namespace luisa {

class Film : public Plugin {

protected:
    Viewport _film_viewport{};
    uint2 _resolution;
    std::shared_ptr<Filter> _filter;
    std::unique_ptr<Buffer<float4>> _accumulation_buffer;
    std::unique_ptr<Kernel> _reset_accumulation_buffer_kernel;
    std::unique_ptr<Kernel> _accumulate_tile_kernel;

public:
    Film(Device *device, const ParameterSet &parameters)
        : Plugin{device},
          _resolution{parameters["resolution"].parse_uint2_or_default(make_uint2(1280, 720))},
          _filter{parameters["filter"].parse_or_null<Filter>()} {
        
        _accumulation_buffer = device->allocate_buffer<float4>(_resolution.x * _resolution.y, BufferStorage::MANAGED);
        _reset_accumulation_buffer_kernel = device->load_kernel("film::reset_accumulation_buffer");
        _accumulate_tile_kernel = device->load_kernel("film::accumulate_tile");
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
    
    virtual void accumulate_tile(KernelDispatcher &dispatch, BufferView<float3> color_buffer, Viewport tile_viewport) {
        dispatch(*_accumulate_tile_kernel, tile_viewport.size.x * tile_viewport.size.y, [&](KernelArgumentEncoder &encode) {
            encode("ray_color_buffer", color_buffer);
            encode("accumulation_buffer", *_accumulation_buffer);
            encode("uniforms", film::AccumulateTileKernelUniforms{tile_viewport, _resolution});
        });
    }
    
    virtual void postprocess(KernelDispatcher &dispatch) = 0;
    virtual void save(std::string_view filename) = 0;
    [[nodiscard]] Filter *filter() noexcept { return _filter.get(); }
    [[nodiscard]] BufferView<float4> accumulation_buffer() noexcept { return _accumulation_buffer->view(); }
    [[nodiscard]] uint2 resolution() noexcept { return _resolution; }
    
};

}

#endif

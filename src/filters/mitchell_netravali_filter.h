//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>
#include <core/mathematics.h>
#include <core/colorspaces.h>
#include <core/viewport.h>

namespace luisa::filter::mitchell_netravali {

LUISA_DEVICE_CALLABLE inline float mitchell_netravali_1d(float x, float b, float c) noexcept {
    x = min(abs(2 * x), 2.0f);
    auto xx = x * x;
    return (1.0f / 6.0f) *
           (x > 1 ?
            ((-b - 6 * c) * xx + (6 * b + 30 * c) * x + (-12 * b - 48 * c)) * x + (8 * b + 24 * c) :
            ((12 - 9 * b - 6 * c) * xx + (-18 + 12 * b + 6 * c) * x) * x + (6 - 2 * b));
}

struct ApplyAndAccumulateKernelUniforms {
    Viewport filter_viewport;
    Viewport tile_viewport;
    uint2 film_resolution;
    float radius;
    float b;
    float c;
};

// todo: viewport
LUISA_DEVICE_CALLABLE inline void apply_and_accumulate(
    LUISA_DEVICE_SPACE const float3 *ray_color_buffer,
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    LUISA_UNIFORM_SPACE ApplyAndAccumulateKernelUniforms &uniforms,
    uint tid) {
    
    if (tid < uniforms.filter_viewport.size.x * uniforms.filter_viewport.size.y) {
        
        auto raster = uniforms.filter_viewport.origin + make_uint2(tid % uniforms.filter_viewport.size.x, tid / uniforms.filter_viewport.size.x);
        auto pixel = make_float2(raster) + make_float2(0.5f);
        
        auto xy_min = make_uint2(max(pixel - make_float2(uniforms.radius), make_float2(uniforms.tile_viewport.origin)));
        auto xy_max = make_uint2(min(pixel + make_float2(uniforms.radius), make_float2(uniforms.tile_viewport.origin + uniforms.tile_viewport.size) - make_float2(1.0f)));
        auto inv_radius = 1.0f / uniforms.radius;
        auto value = make_float4();
        for (auto y = xy_min.y; y <= xy_max.y; y++) {
            for (auto x = xy_min.x; x <= xy_max.x; x++) {
                auto raster_in_tile = make_uint2(x, y) - uniforms.tile_viewport.origin;
                auto index_in_tile = raster_in_tile.y * uniforms.tile_viewport.size.x + raster_in_tile.x;
                auto d = ray_pixel_buffer[index_in_tile] - pixel;
                auto wx = mitchell_netravali_1d(d.x * inv_radius, uniforms.b, uniforms.c);
                auto wy = mitchell_netravali_1d(d.y * inv_radius, uniforms.b, uniforms.c);
                auto weight = wx * wy;
                value += make_float4(ray_color_buffer[index_in_tile] * weight, weight);
            }
        }
        accumulation_buffer[raster.y * uniforms.film_resolution.x + raster.x] += value;
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/filter.h>
#include <core/film.h>

namespace luisa {

class MitchellNetravaliFilter : public Filter {

protected:
    float _b;
    float _c;
    std::unique_ptr<Kernel> _apply_and_accumulate_kernel;

public:
    MitchellNetravaliFilter(Device *device, const ParameterSet &parameters);
    
    void apply_and_accumulate(KernelDispatcher &dispatch,
                              uint2 film_resolution,
                              Viewport film_viewport,
                              Viewport tile_viewport,
                              BufferView<float2> pixel_buffer,
                              BufferView<float3> color_buffer,
                              BufferView<float4> accumulation_buffer) override;
};

LUISA_REGISTER_NODE_CREATOR("MitchellNetravali", MitchellNetravaliFilter)

}

#endif

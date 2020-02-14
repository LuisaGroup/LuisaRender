//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>
#include <core/mathematics.h>
#include <core/color_spaces.h>

namespace luisa::mitchell_netravali_filter {

struct ApplyKernelUniforms {
    uint2 resolution;
    float radius;
    float b;
    float c;
};

LUISA_DEVICE_CALLABLE inline float mitchell_netravali_1d(float x, float b, float c) noexcept {
    x = min(abs(2 * x), 2.0f);
    auto xx = x * x;
    return (1.0f / 6.0f) *
           (x > 1 ?
            ((-b - 6 * c) * xx + (6 * b + 30 * c) * x + (-12 * b - 48 * c)) * x + (8 * b + 24 * c) :
            ((12 - 9 * b - 6 * c) * xx + (-18 + 12 * b + 6 * c) * x) * x + (6 - 2 * b));
}

LUISA_DEVICE_CALLABLE inline void apply(
    LUISA_DEVICE_SPACE const float3 *ray_radiance_buffer,
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE float4 *frame,
    LUISA_UNIFORM_SPACE ApplyKernelUniforms &uniforms,
    uint2 tid) {
    
    if (tid.x < uniforms.resolution.x && tid.y < uniforms.resolution.y) {
        auto index = tid.y * uniforms.resolution.x + tid.x;
        auto pixel = ray_pixel_buffer[index];
        auto x_min = static_cast<uint>(max(0.5f, floor(pixel.x - uniforms.radius)));
        auto x_max = static_cast<uint>(min(uniforms.resolution.x - 0.5f, ceil(pixel.x + uniforms.radius)));
        auto y_min = static_cast<uint>(max(0.5f, floor(pixel.y - uniforms.radius)));
        auto y_max = static_cast<uint>(min(uniforms.resolution.y - 0.5f, ceil(pixel.y + uniforms.radius)));
        auto inv_radius = 1.0f / uniforms.radius;
        auto value = make_float4();
        for (auto y = y_min; y <= y_max; y++) {
            for (auto x = x_min; x <= x_max; x++) {
                auto wx = mitchell_netravali_1d((pixel.x - (x + 0.5f)) * inv_radius, uniforms.b, uniforms.c);
                auto wy = mitchell_netravali_1d((pixel.y - (y + 0.5f)) * inv_radius, uniforms.b, uniforms.c);
                auto weight = wx * wy;
                value += make_float4(ray_radiance_buffer[y * uniforms.resolution.x + x] * weight, weight);
            }
        }
        frame[index] = value;
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
    std::unique_ptr<Kernel> _apply_kernel;

public:
    MitchellNetravaliFilter(Device *device, const ParameterSet &parameters);
    void apply(KernelDispatcher &dispatch, BufferView<float2> pixel_buffer, BufferView<float3> radiance_buffer, BufferView<float4> frame, uint2 film_resolution) override;
};

LUISA_REGISTER_NODE_CREATOR("MitchellNetravali", MitchellNetravaliFilter)

}

#endif

//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "viewport.h"
#include "mathematics.h"
#include "sampling.h"

namespace luisa::filter::separable {

LUISA_CONSTANT_SPACE constexpr auto TABLE_SIZE = 64u;

struct LUT {
    alignas(16) float w[TABLE_SIZE];
    alignas(16) float cdf[TABLE_SIZE];
};

struct ImportanceSamplePixelsKernelUniforms {
    Viewport tile;
    float radius;
    float scale;
};

LUISA_DEVICE_CALLABLE inline float2 sample_1d(float u, LUISA_UNIFORM_SPACE LUT &lut) noexcept {
    
    auto p = 0u;
    for (auto count = static_cast<int>(TABLE_SIZE); count > 0;) {
        auto step = count / 2;
        auto mid = p + step;
        if (lut.cdf[mid] < u) {
            p = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    
    constexpr auto inv_table_size = 1.0f / static_cast<float>(TABLE_SIZE);
    
    auto lb = math::clamp(p, 0u, TABLE_SIZE - 1u);
    auto cdf_lower = lut.cdf[lb];
    auto cdf_upper = (lb == TABLE_SIZE - 1u) ? 1.0f : lut.cdf[lb + 1u];
    auto offset = math::clamp((static_cast<float>(lb) + (u - cdf_lower) / (cdf_upper - cdf_lower)) * inv_table_size, 0.0f, 1.0f);
    
    constexpr auto weight_table_size_float = static_cast<float>(TABLE_SIZE);
    auto index_w = offset * weight_table_size_float;
    auto index_w_lower = math::floor(index_w);
    auto index_w_upper = math::ceil(index_w);
    auto w = lerp(
        lut.w[static_cast<uint>(index_w_lower)],
        index_w_upper >= weight_table_size_float ? 0.0f : lut.w[static_cast<uint>(index_w_upper)],
        index_w - index_w_lower);
    
    return make_float2(offset * 2.0f - 1.0f, w >= 0.0f ? 1.0f : -1.0f);
}

LUISA_DEVICE_CALLABLE inline void importance_sample_pixels(
    LUISA_DEVICE_SPACE const float2 *random_buffer,
    LUISA_DEVICE_SPACE float2 *pixel_location_buffer,
    LUISA_DEVICE_SPACE float3 *pixel_weight_buffer,
    LUISA_UNIFORM_SPACE LUT &lut,
    LUISA_UNIFORM_SPACE ImportanceSamplePixelsKernelUniforms &uniforms,
    uint tid) {
    
    if (tid < uniforms.tile.size.x * uniforms.tile.size.y) {
        auto u = random_buffer[tid];
        auto x_and_wx = sample_1d(u.x, lut);
        auto y_and_wy = sample_1d(u.y, lut);
        pixel_location_buffer[tid] = make_float2(
            static_cast<float>(tid % uniforms.tile.size.x + uniforms.tile.origin.y) + 0.5f + x_and_wx.x * uniforms.radius,
            static_cast<float>(tid / uniforms.tile.size.x + uniforms.tile.origin.y) + 0.5f + y_and_wy.x * uniforms.radius);
        pixel_weight_buffer[tid] = make_float3(x_and_wx.y * y_and_wy.y * uniforms.scale);
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "node.h"
#include "kernel.h"
#include "ray.h"
#include "parser.h"
#include "sampler.h"

namespace luisa {

class Filter : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Filter);

protected:
    float _radius;

public:
    Filter(Device *device, const ParameterSet &parameters)
        : Node{device}, _radius{parameters["radius"].parse_float_or_default(1.0f)} {}
    
    [[nodiscard]] float radius() const noexcept { return _radius; }
    
    virtual void importance_sample_pixels(KernelDispatcher &dispatch,
                                          Viewport tile_viewport,
                                          Sampler &sampler,
                                          BufferView<float2> pixel_location_buffer,
                                          BufferView<float3> pixel_weight_buffer) = 0;
    
};

class SeparableFilter : public Filter {

protected:
    filter::separable::LUT _lut{};
    std::unique_ptr<Kernel> _importance_sample_pixels_kernel;
    float _scale{1.0f};
    bool _lut_computed{false};

protected:
    // Filter 1D weight function, offset is in range [-radius, radius)
    [[nodiscard]] virtual float _weight(float offset) const noexcept = 0;
    
    virtual void _compute_lut_if_necessary();

public:
    SeparableFilter(Device *device, const ParameterSet &parameters);
    void importance_sample_pixels(KernelDispatcher &dispatch,
                                  Viewport tile_viewport,
                                  Sampler &sampler,
                                  BufferView<float2> pixel_location_buffer,
                                  BufferView<float3> pixel_weight_buffer) override;
};
    
}

#endif

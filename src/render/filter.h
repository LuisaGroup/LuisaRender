//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/mathematics.h>

#include "viewport.h"
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

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "core/plugin.h"
#include <compute/kernel.h>
#include "ray.h"
#include "core/parser.h"
#include "sampler.h"

namespace luisa {

class Filter : public Plugin {

protected:
    float _radius;

public:
    Filter(Device *device, const ParameterSet &parameters)
        : Plugin{device}, _radius{parameters["radius"].parse_float_or_default(1.0f)} {}
    
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
    [[nodiscard]] virtual float _weight_1d(float offset) const noexcept = 0;
    
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

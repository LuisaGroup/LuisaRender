//
// Created by Mike Smith on 2020/9/13.
//

#pragma once

#include <compute/pipeline.h>
#include <render/plugin.h>
#include <render/parser.h>
#include <render/sampler.h>
#include <compute/dsl.h>

namespace luisa::render {

using compute::Device;
using compute::Pipeline;
using compute::KernelView;

class Filter : public Plugin {

private:
    float _radius;

public:
    Filter(Device *device, const ParameterSet &params)
        : Plugin{device, params},
          _radius{params["radius"].parse_float_or_default(1.0f)} {}
    
    [[nodiscard]] float radius() const noexcept { return _radius; }
    
    // (position, weight)
    [[nodiscard]] virtual std::pair<Expr<float2>, Expr<float>> importance_sample_pixel_position(Expr<uint2> p, Expr<float2> u) = 0;
};

class SeparableFilter : public Filter {

public:
    static constexpr auto lookup_table_size = 64u;

private:
    std::array<float, lookup_table_size> _weight_table{};
    std::array<float, lookup_table_size> _cdf_table{};
    float _scale{};
    bool _table_generated{false};
    
    // Filter 1D weight function, offset is in range [-radius, radius)
    [[nodiscard]] virtual float _weight_1d(float offset) const noexcept = 0;

public:
    SeparableFilter(Device *device, const ParameterSet &params)
        : Filter{device, params} {}
        
    // (position, weight)
    std::pair<Expr<float2>, Expr<float>> importance_sample_pixel_position(Expr<uint2> p, Expr<float2> u) override;
};

}

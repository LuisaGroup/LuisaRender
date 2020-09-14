//
// Created by Mike Smith on 2020/9/13.
//

#pragma once

#include <compute/pipeline.h>
#include <render/plugin.h>
#include <render/parser.h>
#include <render/sampler.h>

namespace luisa::render {

using compute::Device;
using compute::Pipeline;
using compute::KernelView;

class Filter : public Plugin {

private:
    float _radius;

private:
    virtual void _importance_sample_pixel_positions(
        Pipeline &pipeline,
        Sampler &sampler,
        BufferView<float2> &position_buffer,
        BufferView<float> &weight_buffer) = 0;

public:
    Filter(Device *device, const ParameterSet &params)
        : Plugin{device, params},
          _radius{params["radius"].parse_float_or_default(1.0f)} {}
    
    [[nodiscard]] float radius() const noexcept { return _radius; }
    
    [[nodiscard]] auto importance_sample_pixel_positions(Sampler &sampler, BufferView<float2> &position_buffer, BufferView<float> &weight_buffer) {
        return [this, &sampler, &position_buffer, &weight_buffer](Pipeline &pipeline) {
            _importance_sample_pixel_positions(pipeline, sampler, position_buffer, weight_buffer);
        };
    }
};

class SeparableFilter : public Filter {

public:
    static constexpr auto lookup_table_size = 64u;

private:
    // Filter 1D weight function, offset is in range [-radius, radius)
    [[nodiscard]] virtual float _weight_1d(float offset) const noexcept = 0;
    void _importance_sample_pixel_positions(Pipeline &pipeline, Sampler &sampler, BufferView<float2> &position_buffer, BufferView<float> &weight_buffer) override;

public:
    SeparableFilter(Device *device, const ParameterSet &params)
        : Filter{device, params} {}
};

}

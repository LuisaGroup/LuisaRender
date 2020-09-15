//
// Created by Mike Smith on 2020/9/13.
//

#pragma once

#include <compute/dsl.h>
#include <compute/pipeline.h>
#include <render/parser.h>
#include <render/plugin.h>

namespace luisa::render {

using compute::TextureView;
using compute::BufferView;
using compute::Pipeline;
using compute::dsl::Expr;

class Sampler : public Plugin {

private:
    uint _spp;
    uint _current_frame_index{0u};

private:
    virtual void _prepare_for_next_frame(Pipeline &pipeline) = 0;
    virtual void _reset(Pipeline &pipeline, uint2 resolution) = 0;

public:
    Sampler(Device *device, const ParameterSet &params)
        : Plugin{device, params},
          _spp{params["spp"].parse_uint_or_default(1024u)} {}
    
    [[nodiscard]] uint spp() const noexcept { return _spp; }
    [[nodiscard]] uint current_frame_index() const noexcept { return _current_frame_index; }
    
    [[nodiscard]] virtual Expr<float> generate_1d_sample(Expr<uint> pixel_index) = 0;
    [[nodiscard]] virtual Expr<float2> generate_2d_sample(Expr<uint> pixel_index) = 0;
    [[nodiscard]] virtual Expr<float3> generate_3d_sample(Expr<uint> pixel_index) = 0;
    [[nodiscard]] virtual Expr<float4> generate_4d_sample(Expr<uint> pixel_index) = 0;
    
    [[nodiscard]] auto prepare_for_next_frame() {
        return [this](Pipeline &pipeline) {
            pipeline << [this] {
                _current_frame_index++;
                LUISA_WARNING_IF_NOT(_current_frame_index <= _spp, "Current frame index ", _current_frame_index, " exceeds samples per pixel: ", _spp);
            };
            _prepare_for_next_frame(pipeline);
        };
    }
    
    [[nodiscard]] auto reset(uint2 resolution) {
        return [this, resolution](Pipeline &pipeline) {
            pipeline << [this] { _current_frame_index = 0u; };
            _reset(pipeline, resolution);
        };
    }
};

}

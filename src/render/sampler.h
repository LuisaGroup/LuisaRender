//
// Created by Mike Smith on 2020/9/13.
//

#pragma once

#include <compute/pipeline.h>
#include <render/parser.h>
#include <render/plugin.h>

namespace luisa::render {

using compute::TextureView;
using compute::BufferView;
using compute::Pipeline;

class Sampler : public Plugin {

private:
    uint _spp;
    uint _current_frame_index{0u};
    uint _current_sample_index{0u};
    TextureView _samples;

private:
    virtual void _generate_samples(Pipeline &pipeline, TextureView &samples, uint num_dims) = 0;
    virtual void _prepare_for_next_frame(Pipeline &pipeline) = 0;
    virtual void _reset(Pipeline &pipeline, uint2 resolution) = 0;
    
    [[nodiscard]] virtual uint _max_sample_dimensions() = 0;

public:
    Sampler(Device *device, const ParameterSet &params)
        : Plugin{device, params},
          _spp{params["spp"].parse_uint_or_default(1024u)} {}
    
    [[nodiscard]] uint spp() const noexcept { return _spp; }
    [[nodiscard]] uint current_frame_index() const noexcept { return _current_frame_index; }
    [[nodiscard]] uint current_sample_index() const noexcept { return _current_sample_index; }
    
    [[nodiscard]] const TextureView &sample_texture() const noexcept { return _samples; }
    
    [[nodiscard]] auto generate_samples(uint num_dims) {
        return [this, num_dims](Pipeline &pipeline) {
            pipeline << [this, num_dims] {
                LUISA_EXCEPTION_IF_NOT(num_dims <= 4u, "Cannot generate samples beyond 4D, requested: ", num_dims);
                LUISA_EXCEPTION_IF_NOT(_max_sample_dimensions() == 0u || _current_sample_index + num_dims <= _max_sample_dimensions(),
                                       "Current sample dimension exceeds max sample dimension: ", _max_sample_dimensions());
            };
            _generate_samples(pipeline, _samples, num_dims);
            pipeline << [this, num_dims] { _current_sample_index += num_dims; };
        };
    }
    
    [[nodiscard]] auto prepare_for_next_frame() {
        return [this](Pipeline &pipeline) {
            pipeline << [this] {
                _current_sample_index = 0u;
                _current_frame_index++;
                LUISA_WARNING_IF_NOT(_current_frame_index <= _spp, "Current frame index ", _current_frame_index, " exceeds samples per pixel: ", _spp);
            };
            _prepare_for_next_frame(pipeline);
        };
    }
    
    [[nodiscard]] auto reset(uint2 resolution) {
        return [this, resolution](Pipeline &pipeline) {
            pipeline << [this, resolution] {
                if (_samples.empty() || _samples.width() != resolution.x || _samples.height() != resolution.y) {
                    _samples = device()->allocate_texture<float4>(resolution.x, resolution.y);
                }
                _current_frame_index = 0u;
                _current_sample_index = 0u;
            };
            _reset(pipeline, resolution);
        };
    }
};

}

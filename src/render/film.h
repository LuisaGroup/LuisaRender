//
// Created by Mike Smith on 2020/9/13.
//

#pragma once

#include <compute/pipeline.h>
#include <render/plugin.h>
#include <render/parser.h>

namespace luisa::render {

using compute::TextureView;
using compute::BufferView;
using compute::Pipeline;

class Film : public Plugin {

private:
    uint2 _resolution;

protected:
    virtual void _clear(Pipeline &pipeline) = 0;
    virtual void _accumulate_frame(Pipeline &pipeline, const BufferView<float3> &radiance_buffer, const BufferView<float> &weight_buffer) = 0;
    virtual void _postprocess(Pipeline &pipeline) = 0;

public:
    Film(Device *device, const ParameterSet &params)
        : Plugin{device, params},
          _resolution{params["resolution"].parse_uint2_or_default(make_uint2(1280u, 720u))} {}
    
    [[nodiscard]] uint2 resolution() const noexcept { return _resolution; }
    
    [[nodiscard]] auto clear() {
        return [this](Pipeline &pipeline) { _clear(pipeline); };
    }
    
    [[nodiscard]] auto accumulate_frame(const BufferView<float3> &radiance_buffer, const BufferView<float> &weight_buffer) {
        return [this, &radiance_buffer, &weight_buffer](Pipeline &pipeline) {
            _accumulate_frame(pipeline, radiance_buffer, weight_buffer);
        };
    }
    
    [[nodiscard]] auto postprocess() {
        return [this](Pipeline &pipeline) { _postprocess(pipeline); };
    }
};

}

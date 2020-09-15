//
// Created by Mike Smith on 2020/9/14.
//

#pragma once

#include <render/plugin.h>
#include <render/parser.h>
#include <render/sampler.h>
#include <render/scene.h>

namespace luisa::render {

class Integrator : public Plugin {

private:
    BufferView<float3> _radiance_buffer;

private:
    virtual void _render_frame(Pipeline &pipeline,
                               Scene &scene,
                               Sampler &sampler,
                               BufferView<Ray> &ray_buffer,
                               BufferView<float3> &throughput_buffer,
                               BufferView<float3> &radiance_buffer) = 0;

public:
    Integrator(Device *device, const ParameterSet &params) noexcept
        : Plugin{device, params} {}
    
    [[nodiscard]] const BufferView<float3> &radiance_buffer() const noexcept { return _radiance_buffer; }
    
    [[nodiscard]] auto render_frame(Scene &scene, Sampler &sampler, BufferView<Ray> &ray_buffer, BufferView<float3> &throughput_buffer) {
        if (_radiance_buffer.empty()) {
            _radiance_buffer = device()->allocate_buffer<float3>(ray_buffer.size());
        }
        return [this, &scene, &sampler, &ray_buffer, &throughput_buffer](Pipeline &pipeline) {
            _render_frame(pipeline, scene, sampler, ray_buffer, throughput_buffer, _radiance_buffer);
        };
    }
};

}

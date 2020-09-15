//
// Created by Mike Smith on 2020/9/15.
//

#include <compute/dsl.h>
#include <render/camera.h>
#include <render/integrator.h>

namespace luisa::render::integrator {

using namespace compute;
using namespace compute::dsl;

class NormalVisualizer : public Integrator {

private:
    void _render_frame(Pipeline &pipeline, Scene &scene, Sampler &sampler,
                       BufferView<Ray> &ray_buffer, BufferView<float3> &throughput_buffer, BufferView<float3> &radiance_buffer) override {
        
    }

public:
    NormalVisualizer(Device *device, const ParameterSet &params)
        : Integrator{device, params} {}
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::integrator::NormalVisualizer)

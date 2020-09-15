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
    KernelView _normal_to_color_kernel;

private:
    void _render_frame(Pipeline &pipeline, Scene &scene, Sampler &sampler,
                       BufferView<Ray> &ray_buffer, BufferView<float3> &throughput_buffer, BufferView<float3> &radiance_buffer) override {
        
        pipeline << scene.intersect_closest(ray_buffer);
        auto pixel_count = static_cast<uint>(ray_buffer.size());
        static constexpr auto threadgroup_size = 256u;
        if (_normal_to_color_kernel.empty()) {
            _normal_to_color_kernel = device()->compile_kernel("normal_visualizer_colorize_normal", [&] {
                auto tid = thread_id();
                If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                    Var valid = scene.interaction_buffers().valid[tid];
                    Var normal = scene.interaction_buffers().ns[tid];
                    radiance_buffer[tid] = select(valid, normal * 0.5f + 0.5f, make_float3(0.0f));
                };
            });
        }
        pipeline << _normal_to_color_kernel.parallelize(pixel_count, threadgroup_size);
    }

public:
    NormalVisualizer(Device *device, const ParameterSet &params)
        : Integrator{device, params} {}
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::integrator::NormalVisualizer)

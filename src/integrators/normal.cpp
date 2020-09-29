//
// Created by Mike Smith on 2020/9/15.
//

#include <compute/dsl_syntax.h>
#include <render/camera.h>
#include <render/integrator.h>

namespace luisa::render::integrator {

using namespace compute;
using namespace compute::dsl;

class NormalVisualizer : public Integrator {

private:
    BufferView<ClosestHit> _hit_buffer;

private:
    void _render_frame(Pipeline &pipeline, Scene &scene, Sampler &sampler,
                       BufferView<Ray> &ray_buffer, BufferView<float3> &throughput_buffer, BufferView<float3> &radiance_buffer) override {
        
        auto pixel_count = static_cast<uint>(ray_buffer.size());
        static constexpr auto threadgroup_size = 256u;
        
        pipeline << scene.intersect_closest(ray_buffer, _hit_buffer)
                 << device()->compile_kernel("normal_visualizer_colorize_normal", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                         auto interaction = scene.evaluate_interaction(ray_buffer[tid], _hit_buffer[tid], Interaction::COMPONENT_NS | Interaction::COMPONENT_MISS);
                         Var miss = *interaction.miss;
                         Var normal = *interaction.ns;
                         Var throughput = throughput_buffer[tid];
                         radiance_buffer[tid] = select(miss, make_float3(0.5f), throughput * normal * 0.5f + 0.5f);
                     };
                 }).parallelize(pixel_count, threadgroup_size);
    }

public:
    NormalVisualizer(Device *device, const ParameterSet &params)
        : Integrator{device, params} {}
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::integrator::NormalVisualizer)

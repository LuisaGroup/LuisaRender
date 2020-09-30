//
// Created by Mike Smith on 2020/9/16.
//

#include <compute/dsl_syntax.h>
#include <render/integrator.h>
#include <render/sampling.h>

namespace luisa::render::integrator {

using namespace compute;
using namespace compute::dsl;

class AmbientOcclusion : public Integrator {

private:
    BufferView<AnyHit> _any_hit_buffer;
    BufferView<ClosestHit> _closest_hit_buffer;

private:
    void _render_frame(Pipeline &pipeline,
                       Scene &scene,
                       Sampler &sampler,
                       BufferView<Ray> &ray_buffer,
                       BufferView<float3> &throughput_buffer,
                       BufferView<float3> &radiance_buffer) override {
        
        auto pixel_count = static_cast<uint>(ray_buffer.size());
        static constexpr auto threadgroup_size = 256u;
        
        if (_any_hit_buffer.size() < pixel_count) {
            _any_hit_buffer = device()->allocate_buffer<AnyHit>(pixel_count);
            _closest_hit_buffer = device()->allocate_buffer<ClosestHit>(pixel_count);
        }
        
        pipeline << scene.intersect_closest(ray_buffer, _closest_hit_buffer)
                 << device()->compile_kernel("ao_compute_scattering", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                
                         auto interaction = scene.evaluate_interaction(
                             ray_buffer[tid], _closest_hit_buffer[tid], 0.0f,
                             Interaction::COMPONENT_MISS |
                             Interaction::COMPONENT_NG |
                             Interaction::COMPONENT_NS |
                             Interaction::COMPONENT_PI |
                             Interaction::COMPONENT_SHADER |
                             Interaction::COMPONENT_WO);
                
                         Var u = sampler.generate_2d_sample(tid);
                         auto scattering = scene.evaluate_scattering(interaction, make_float3(0.0f), u, SurfaceShader::EVAL_BSDF_SAMPLING);
                         throughput_buffer[tid] *= select(interaction.miss || scattering.sample.pdf == 0.0f,
                                                          make_float3(0.0f),
                                                          scattering.sample.f * abs(dot(scattering.sample.wi, interaction.ns)) / max(scattering.sample.pdf, 1e-3f));
                
                         Var position = offset_ray_origin(interaction.pi, interaction.ng);
                         Var<Ray> shadow_ray;
                         shadow_ray.origin_x = position.x;
                         shadow_ray.origin_y = position.y;
                         shadow_ray.origin_z = position.z;
                         shadow_ray.min_distance = 0.0f;
                         shadow_ray.direction_x = scattering.sample.wi.x;
                         shadow_ray.direction_y = scattering.sample.wi.y;
                         shadow_ray.direction_z = scattering.sample.wi.z;
                         shadow_ray.max_distance = select(interaction.miss, -1.0f, 1e3f);
                         ray_buffer[tid] = shadow_ray;
                     };
                 }).parallelize(pixel_count, threadgroup_size)
                 << scene.intersect_any(ray_buffer, _any_hit_buffer)
                 << device()->compile_kernel("ao_evaluate_shadows", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                         Var its_distance = _any_hit_buffer[tid].distance;
                         Var throughput = throughput_buffer[tid];
                         radiance_buffer[tid] = throughput * select(its_distance <= 0.0f, 1.0f, 0.0f);
                     };
                 }).parallelize(pixel_count, threadgroup_size);
    }

public:
    AmbientOcclusion(Device *d, const ParameterSet &params)
        : Integrator{d, params} {}
    
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::integrator::AmbientOcclusion)

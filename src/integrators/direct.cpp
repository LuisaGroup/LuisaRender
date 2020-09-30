//
// Created by Mike Smith on 2020/9/30.
//

#include <compute/dsl_syntax.h>
#include <render/integrator.h>

namespace luisa::render::integrator {

using namespace compute;
using namespace compute::dsl;

class DirectLighting : public Integrator {

private:
    BufferView<AnyHit> _any_hit_buffer;
    BufferView<ClosestHit> _closest_hit_buffer;
    BufferView<float3> _direct_radiance_buffer;

private:
    void _render_frame(Pipeline &pipeline,
                       Scene &scene,
                       Sampler &sampler,
                       BufferView<Ray> &ray_buffer,
                       BufferView<float3> &throughput_buffer,
                       BufferView<float3> &radiance_buffer) override {
        
        auto pixel_count = static_cast<uint>(ray_buffer.size());
        static constexpr auto threadgroup_size = 1024u;
        
        if (_any_hit_buffer.size() < pixel_count) {
            _any_hit_buffer = device()->allocate_buffer<AnyHit>(pixel_count);
            _closest_hit_buffer = device()->allocate_buffer<ClosestHit>(pixel_count);
            _direct_radiance_buffer = device()->allocate_buffer<float3>(pixel_count);
        }
        
        pipeline << scene.intersect_closest(ray_buffer, _closest_hit_buffer)
                 << device()->compile_kernel("direct_generate_shadow_rays", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                
                         auto u = sampler.generate_3d_sample(tid);
                         auto v = sampler.generate_4d_sample(tid);
                
                         auto interaction = scene.evaluate_interaction(ray_buffer[tid], _closest_hit_buffer[tid], u.x, Interaction::COMPONENT_ALL);
                
                         If (interaction.miss) {
                             _direct_radiance_buffer[tid] = make_float3(0.0f);
                             ray_buffer[tid].max_distance = -1.0f;
                         } Else {
                             auto light_selection = scene.uniform_select_light(u.y, u.z);
                             auto light_sample = scene.uniform_sample_light(light_selection, interaction.pi, make_float2(v.x, v.y));
                             auto scattering = scene.evaluate_scattering(interaction, light_sample.wi, make_float2(v.z, v.w),
                                                                         SurfaceShader::EVAL_EMISSION | SurfaceShader::EVAL_BSDF);
                    
                             Var Le = scattering.emission.L;
                             Var Li = light_sample.Li / max(light_sample.pdf * light_selection.prob, 1e-4f);
                             _direct_radiance_buffer[tid] = Le + Li * scattering.evaluation.f * abs(dot(interaction.ns, light_sample.wi)) /
                                                                 max(scattering.evaluation.pdf, 1e-4f);
                             Var position = offset_ray_origin(interaction.pi, interaction.ng);
                             Var<Ray> shadow_ray;
                             shadow_ray.origin_x = position.x;
                             shadow_ray.origin_y = position.y;
                             shadow_ray.origin_z = position.z;
                             shadow_ray.min_distance = 0.0f;
                             shadow_ray.direction_x = light_sample.wi.x;
                             shadow_ray.direction_y = light_sample.wi.y;
                             shadow_ray.direction_z = light_sample.wi.z;
                             shadow_ray.max_distance = select(all(light_sample.Li == 0.0f), -1.0f, light_sample.distance - 1e-3f);
                             ray_buffer[tid] = shadow_ray;
                         };
                     };
                 }).parallelize(pixel_count, threadgroup_size)
                 << scene.intersect_any(ray_buffer, _any_hit_buffer)
                 << device()->compile_kernel("direct_evaluate_shadows", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                         Var its_distance = _any_hit_buffer[tid].distance;
                         Var throughput = throughput_buffer[tid];
                         radiance_buffer[tid] = throughput * select(its_distance <= 0.0f, _direct_radiance_buffer[tid], make_float3(0.0f));
                     };
                 }).parallelize(pixel_count, threadgroup_size);
    }

public:
    DirectLighting(Device *d, const ParameterSet &params)
        : Integrator{d, params} {}
    
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::integrator::DirectLighting)

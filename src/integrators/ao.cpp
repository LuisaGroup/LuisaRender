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
    BufferView<bool> _miss_buffer;

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
            _miss_buffer = device()->allocate_buffer<bool>(pixel_count);
        }
        
        pipeline << scene.intersect_closest(ray_buffer, _closest_hit_buffer)
                 << device()->compile_kernel("ao_generate_shadow_rays", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                
                         auto interaction = scene.evaluate_interaction(
                             ray_buffer[tid], _closest_hit_buffer[tid],
                             Interaction::COMPONENT_MISS | Interaction::COMPONENT_NG | Interaction::COMPONENT_PI);
                
                         Var normal = *interaction.ng;
                         Var miss = *interaction.miss;
                         Var position = offset_ray_origin(*interaction.pi, normal);
                         
                         _miss_buffer[tid] = miss;
                
                         Var onb = make_onb(normal);
                         Var u = sampler.generate_2d_sample(tid);
                         Var direction_local = cosine_sample_hemisphere(u);
                         Var direction = normalize(transform_to_world(onb, direction_local));
                
                         Var<Ray> shadow_ray;
                         shadow_ray.origin_x = position.x;
                         shadow_ray.origin_y = position.y;
                         shadow_ray.origin_z = position.z;
                         shadow_ray.min_distance = 0.0f;
                         shadow_ray.direction_x = direction.x;
                         shadow_ray.direction_y = direction.y;
                         shadow_ray.direction_z = direction.z;
                         shadow_ray.max_distance = select(miss, -1.0f, 1e3f);
                         ray_buffer[tid] = shadow_ray;
                     };
                 }).parallelize(pixel_count, threadgroup_size)
                 << scene.intersect_any(ray_buffer, _any_hit_buffer)
                 << device()->compile_kernel("ao_evaluate_shadows", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                         Var its_distance = _any_hit_buffer[tid].distance;
                         Var miss = _miss_buffer[tid];
                         Var throughput = throughput_buffer[tid];
                         radiance_buffer[tid] = throughput * select(!miss && its_distance <= 0.0f, 1.0f, 0.0f);
                     };
                 }).parallelize(pixel_count, threadgroup_size);
    }

public:
    AmbientOcclusion(Device *d, const ParameterSet &params)
        : Integrator{d, params} {}
    
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::integrator::AmbientOcclusion)

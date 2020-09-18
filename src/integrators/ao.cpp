//
// Created by Mike Smith on 2020/9/16.
//

#include <compute/dsl.h>
#include <render/integrator.h>
#include <render/sampling.h>

namespace luisa::render::integrator {

using namespace compute;
using namespace compute::dsl;

class AmbientOcclusion : public Integrator {

private:
    void _render_frame(Pipeline &pipeline,
                       Scene &scene,
                       Sampler &sampler,
                       BufferView<Ray> &ray_buffer,
                       BufferView<float3> &throughput_buffer,
                       BufferView<float3> &radiance_buffer) override {
        
        auto pixel_count = static_cast<uint>(ray_buffer.size());
        static constexpr auto threadgroup_size = 256u;
        
        pipeline << scene.intersect_closest(ray_buffer)
                 << device()->compile_kernel("ao_generate_shadow_rays", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                
                         Var valid = scene.interaction_buffers().valid[tid];
                         Var normal = face_forward(scene.interaction_buffers().ng[tid], scene.interaction_buffers().hit_to_ray_origin[tid]);
                         Var position = offset_ray_origin(scene.interaction_buffers().pi[tid], normal);
                
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
                         shadow_ray.max_distance = select(valid, 1e3f, -1.0f);
                         ray_buffer[tid] = shadow_ray;
                     };
                 }).parallelize(pixel_count, threadgroup_size)
                 << scene.intersect_any(ray_buffer)
                 << device()->compile_kernel("ao_evaluate_shadows", [&] {
                     auto tid = thread_id();
                     If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                         Var its_distance = scene.any_hit_buffer()[tid].distance;
                         Var valid = scene.interaction_buffers().valid[tid];
                         Var throughput = throughput_buffer[tid];
                         radiance_buffer[tid] = throughput * select(valid && its_distance <= 0.0f, 1.0f, 0.0f);
                     };
                 }).parallelize(pixel_count, threadgroup_size);
    }

public:
    AmbientOcclusion(Device *d, const ParameterSet &params)
        : Integrator{d, params} {}
    
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::integrator::AmbientOcclusion)

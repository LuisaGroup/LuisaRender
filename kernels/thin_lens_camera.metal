#include "compatibility.h"

#include <core/data_types.h>
#include <core/mathematics.h>
#include <core/ray.h>
#include <core/sampling.h>
#include <core/sampler.h>

#include <cameras/thin_lens_camera.h>

using namespace luisa;

LUISA_KERNEL void thin_lens_camera_generate_rays(
    LUISA_DEVICE_SPACE float3 *ray_throughput_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_DEVICE_SPACE SamplerState *ray_sampler_state_buffer,
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_queue_size,
    LUISA_UNIFORM_SPACE ThinLensCameraGenerateRaysKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_queue_size) {
        
        auto ray_index = ray_queue[tid.x];
        auto pixel = ray_pixel_buffer[ray_index];
        auto p_focal = (make_float2(0.5f) - pixel / make_float2(uniforms.film_resolution)) * (uniforms.focal_plane / uniforms.near_plane);
        auto p_focal_world = p_focal.x * uniforms.camera_left + p_focal.y * uniforms.camera_up + uniforms.focal_plane * uniforms.camera_front;
        
        auto sampler_state = ray_sampler_state_buffer[ray_index];
        auto r1 = sampler_generate_sample(sampler_state);
        auto r2 = sampler_generate_sample(sampler_state);
        auto p_lens = concentric_sample_disk(r1, r2) * uniforms.lens_radius;
        auto p_lens_world = p_lens.x * uniforms.camera_left + p_lens.y * uniforms.camera_up + uniforms.camera_position;
        
        ray_buffer[tid.x] = make_ray(p_lens_world, normalize(p_focal_world - p_lens_world));
        ray_sampler_state_buffer[ray_index] = sampler_state;
        ray_throughput_buffer[ray_index] = make_float3(1.0f);
    }
    
}

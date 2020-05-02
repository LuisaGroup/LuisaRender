//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include <core/data_types.h>
#include <core/ray.h>
#include <core/sampling.h>
#include <core/viewport.h>

namespace luisa::camera::thin_lens {

struct GenerateRaysKernelUniforms {
    float3 camera_position;
    float3 camera_left;
    float3 camera_up;
    float3 camera_front;
    uint2 film_resolution;
    float2 sensor_size;
    float near_plane;
    float focal_plane;
    float lens_radius;
    Viewport tile_viewport;
    float4x4 transform;
};

LUISA_DEVICE_CALLABLE inline void generate_rays(
    LUISA_DEVICE_SPACE const float2 *sample_buffer,
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_UNIFORM_SPACE GenerateRaysKernelUniforms &uniforms,
    uint tid) noexcept {
    
    if (tid < uniforms.tile_viewport.size.x * uniforms.tile_viewport.size.y) {
        
        auto pixel = ray_pixel_buffer[tid];
        auto p_focal = (make_float2(0.5f) - pixel / make_float2(uniforms.film_resolution)) * uniforms.sensor_size * 0.5f * (uniforms.focal_plane / uniforms.near_plane);
        auto p_focal_world = make_float3(uniforms.transform * make_float4(
            p_focal.x * uniforms.camera_left + p_focal.y * uniforms.camera_up + uniforms.focal_plane * uniforms.camera_front + uniforms.camera_position, 1.0f));
    
        auto sample = sample_buffer[tid];
        auto p_lens = concentric_sample_disk(sample.x, sample.y) * uniforms.lens_radius;
        auto p_lens_world = make_float3(uniforms.transform * make_float4(p_lens.x * uniforms.camera_left + p_lens.y * uniforms.camera_up + uniforms.camera_position, 1.0f));
        
        ray_buffer[tid] = make_ray(p_lens_world, normalize(p_focal_world - p_lens_world));
    }
}

}

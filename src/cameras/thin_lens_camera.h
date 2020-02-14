//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include <core/data_types.h>
#include <core/ray.h>
#include <core/sampling.h>

namespace luisa::thin_lens_camera {

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
};

LUISA_DEVICE_CALLABLE inline void thin_lens_camera_generate_rays(
    LUISA_DEVICE_SPACE const float4 *sample_buffer,
    LUISA_DEVICE_SPACE float3 *ray_throughput_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_DEVICE_SPACE float2 *ray_pixel_buffer,
    LUISA_UNIFORM_SPACE GenerateRaysKernelUniforms &uniforms,
    uint tid) {
    
    if (tid < uniforms.film_resolution.x * uniforms.film_resolution.y) {
        
        auto sample = sample_buffer[tid];
        
        auto pixel = make_float2(tid % uniforms.film_resolution.x, tid / uniforms.film_resolution.y) + make_float2(sample);
        auto p_focal = (make_float2(0.5f) - pixel / make_float2(uniforms.film_resolution)) * (uniforms.focal_plane / uniforms.near_plane);
        auto p_focal_world = p_focal.x * uniforms.camera_left + p_focal.y * uniforms.camera_up + uniforms.focal_plane * uniforms.camera_front;
        
        auto p_lens = concentric_sample_disk(sample.z, sample.w) * uniforms.lens_radius;
        auto p_lens_world = p_lens.x * uniforms.camera_left + p_lens.y * uniforms.camera_up + uniforms.camera_position;
        
        ray_buffer[tid] = make_ray(p_lens_world, normalize(p_focal_world - p_lens_world));
        ray_pixel_buffer[tid] = pixel;
        ray_throughput_buffer[tid] = make_float3(1.0f);
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/camera.h>
#include <core/mathematics.h>

namespace luisa {

class ThinLensCamera : public Camera {

protected:
    float3 _position{};
    float3 _front{};
    float3 _up{};
    float3 _left{};
    float _focal_plane_distance{};
    float _near_plane_distance{};
    float _lens_radius{};
    float2 _sensor_size{};
    float2 _effective_sensor_size{};
    
    std::unique_ptr<Kernel> _generate_rays_kernel;

public:
    ThinLensCamera(Device *device, const ParameterSet &parameters);
    void generate_rays(KernelDispatcher &dispatch,
                       BufferView<float4> sample_buffer,
                       BufferView<float2> pixel_buffer,
                       BufferView<Ray> ray_buffer,
                       BufferView<float3> throughput_buffer) override;
    
};

LUISA_REGISTER_NODE_CREATOR("ThinLens", ThinLensCamera)

}

#endif

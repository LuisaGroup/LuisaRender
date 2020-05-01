//
// Created by Mike Smith on 2020/2/20.
//

#pragma once

#include <core/ray.h>
#include <core/mathematics.h>
#include <core/viewport.h>

namespace luisa::camera::pinhole {

struct GenerateRaysKernelUniforms {
    float3 camera_position;
    float3 camera_left;
    float3 camera_up;
    float3 camera_front;
    uint2 film_resolution;
    float2 sensor_size;
    float near_plane;
    Viewport tile_viewport;
    float4x4 transform;
};

LUISA_DEVICE_CALLABLE inline void generate_rays(
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_UNIFORM_SPACE GenerateRaysKernelUniforms &uniforms,
    uint tid) noexcept {
    
    if (tid < uniforms.tile_viewport.size.x * uniforms.tile_viewport.size.y) {
        
        auto pixel = ray_pixel_buffer[tid];
        
        auto p_film = (make_float2(0.5f) - pixel / make_float2(uniforms.film_resolution)) * uniforms.sensor_size * 0.5f;
        auto o_world = make_float3(uniforms.transform * make_float4(uniforms.camera_position, 1.0f));
        auto p_film_world = make_float3(uniforms.transform * make_float4(
            p_film.x * uniforms.camera_left + p_film.y * uniforms.camera_up + uniforms.near_plane * uniforms.camera_front + uniforms.camera_position, 1.0f));
        
        ray_buffer[tid] = make_ray(o_world, normalize(p_film_world - o_world));
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/camera.h>

namespace luisa {

class PinholeCamera : public Camera {

protected:
    float3 _position;
    float3 _front{};
    float3 _up{};
    float3 _left{};
    float2 _sensor_size{};
    float _near_plane;
    
    std::unique_ptr<Kernel> _generate_rays_kernel;
    
protected:
    void _generate_rays(KernelDispatcher &dispatch,
                       Sampler &sampler,
                       Viewport tile_viewport,
                       BufferView<float2> pixel_buffer,
                       BufferView<Ray> ray_buffer,
                       BufferView<float3> throughput_buffer) override;

public:
    PinholeCamera(Device *device, const ParameterSet &parameter_set);
};

}

#endif
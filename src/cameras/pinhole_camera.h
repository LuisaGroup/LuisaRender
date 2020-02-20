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
};

LUISA_DEVICE_CALLABLE inline void generate_rays(
    LUISA_DEVICE_SPACE const float4 *sample_buffer,
    LUISA_DEVICE_SPACE float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_DEVICE_SPACE float3 *ray_throughput_buffer,
    LUISA_UNIFORM_SPACE GenerateRaysKernelUniforms &uniforms,
    uint tid) noexcept {
    
    if (tid < uniforms.tile_viewport.size.x * uniforms.tile_viewport.size.y) {
        
        auto pixel = make_float2(sample_buffer[tid]) + make_float2(uniforms.tile_viewport.origin)
                     + make_float2(make_uint2(tid % uniforms.tile_viewport.size.x, tid / uniforms.tile_viewport.size.x));
        
        auto p_film = (make_float2(0.5f) - pixel / make_float2(uniforms.film_resolution)) * uniforms.sensor_size * 0.5f;
        auto d = p_film.x * uniforms.camera_left + p_film.y * uniforms.camera_up + uniforms.near_plane * uniforms.camera_front;
        
        ray_buffer[tid] = make_ray(uniforms.camera_position, normalize(d));
        
        ray_pixel_buffer[tid] = pixel;
        ray_throughput_buffer[tid] = make_float3(1.0f);
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

public:
    PinholeCamera(Device *device, const ParameterSet &parameter_set);
    void generate_rays(KernelDispatcher &dispatch,
                       Sampler &sampler,
                       Viewport tile_viewport,
                       BufferView<float2> pixel_buffer,
                       BufferView<Ray> ray_buffer,
                       BufferView<float3> throughput_buffer) override;
};

LUISA_REGISTER_NODE_CREATOR("Pinhole", PinholeCamera)

}

#endif
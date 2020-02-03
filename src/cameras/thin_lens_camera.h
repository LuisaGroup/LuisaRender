//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct ThinLensCameraGenerateRaysKernelUniforms {
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

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/camera.h>
#include <core/mathematics.h>

namespace luisa {

class ThinLensCamera : public Camera {

protected:
    float3 _position;
    float3 _front{};
    float3 _up{};
    float3 _left{};
    float _focal_plane_distance{};
    float _f_number;
    float _focal_length;  // uint: mm
    float2 _sensor_size;  // unit: mm
    float2 _effective_sensor_size{};
    
    std::unique_ptr<Kernel> _generate_rays_kernel;

public:
    ThinLensCamera(Device *device, const ParameterSet &params) : Camera{device, params} {}
    ThinLensCamera(Device *device, std::shared_ptr<Film> film,
                   float3 position, float3 target, float3 up = make_float3(0.0f, 1.0f, 0.0f),
                   float f_number = 1.2f, float focal_length = 50.0f, float2 sensor_size = make_float2(36.0f, 24.0f))
        : Camera{device, std::move(film)}, _position{position}, _f_number{f_number}, _focal_length{focal_length}, _sensor_size{sensor_size} {
        
        auto forward = target - position;
        _front = normalize(forward);
        _left = normalize(cross(up, _front));
        _up = normalize(cross(_front, _left));
        _focal_plane_distance = length(forward);
        
        _generate_rays_kernel = device->create_kernel("thin_lens_camera_generate_rays");
        auto film_resolution = make_float2(_film->resolution());
        auto film_aspect = film_resolution.x / film_resolution.y;
        auto sensor_aspect = _sensor_size.x / _sensor_size.y;
        _effective_sensor_size = sensor_aspect < film_aspect ?
                                 make_float2(_sensor_size.x, _sensor_size.x / film_aspect) :
                                 make_float2(_sensor_size.y * film_aspect, _sensor_size.y);
    }
    
    void generate_rays(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue) override;
    
};

LUISA_REGISTER_NODE_CREATOR("ThinLens", ThinLensCamera)

}

#endif

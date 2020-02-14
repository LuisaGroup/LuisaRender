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
                       BufferView<float2> pixel_buffer,
                       BufferView<SamplerState> sampler_state_buffer,
                       BufferView<float3> throughput_buffer,
                       BufferView<uint> ray_queue_buffer,
                       BufferView<uint> ray_queue_size_buffer,
                       BufferView<Ray> ray_buffer) override;
    
};

LUISA_REGISTER_NODE_CREATOR("ThinLens", ThinLensCamera)

}

#endif

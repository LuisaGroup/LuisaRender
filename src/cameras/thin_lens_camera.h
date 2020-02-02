//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct ThinLensCameraGenerateRaysKernelUniforms {
    float3 position;
    float3 axis_x;
    float3 axis_y;
    float3 axis_z;
    float2 frame_size;
    float2 sensor_size;
    float near_plane;
    float focal_plane;
    float lens_radius;
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/camera.h>

namespace luisa {

class ThinLensCamera : public Camera {

private:
    float _fov;

public:

};

}

#endif

//
// Created by Mike Smith on 2020/2/20.
//

#pragma once

#include <render/ray.h>
#include <core/mathematics.h>
#include <render/viewport.h>

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

}

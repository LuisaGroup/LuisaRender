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

}

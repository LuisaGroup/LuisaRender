//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct CameraData {
    float3 position;
    float3 front;
    float3 left;
    float3 up;
    float near_plane;
    float fov;
    float aperture;
    float focal_distance;
};

}

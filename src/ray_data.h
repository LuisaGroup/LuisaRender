//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct Ray {
    packed_float3 origin;
    float min_distance;
    packed_float3 direction;
    float max_distance;
};

struct GatherRayData {
    float3 radiance;
    float2 pixel;
    float2 padding{};
};

enum struct RayState : uint8_t {
    UNINITIALIZED,
    GENERATED,
    TRACED,
    SHADED,
    FINISHED
};

}
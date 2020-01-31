//
// Created by Mike Smith on 2020/1/31.
//

#pragma once

#include "data_types.h"

namespace luisa {

struct Ray {
    packed_float3 origin;
    float min_distance;
    packed_float3 direction;
    float max_distance;
};

enum struct RayState : uint8_t {
    UNINITIALIZED,
    GENERATED,
    TRACED,
    SHADED,
    FINISHED
};

}

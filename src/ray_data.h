//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#include "compatibility.h"

struct Ray {
    PackedVec3f origin;
    float min_distance;
    PackedVec3f direction;
    float max_distance;
};

struct GatherRayData {
    Vec3f radiance;
    Vec2f pixel;
    Vec2f padding{};
};

enum struct RayState : uint8_t {
    UNINITIALIZED,
    GENERATED,
    TRACED,
    SHADED,
    FINISHED
};

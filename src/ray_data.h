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

struct LightSample {
    PackedVec3f radiance;
    float pdf;
    PackedVec3f direction;
    float distance;
};

struct GatherRayData {
    Vec3f radiance;
    Vec2f pixel;
    Vec2f padding{};
};

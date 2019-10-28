//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#include "compatibility.h"

struct RayData {
    PackedVec3f origin;
    float min_distance;
    PackedVec3f direction;
    float max_distance;
    PackedVec3f throughput;
    uint seed;
    PackedVec3f radiance;
    uint depth;
    Vec2f pixel;
    float pdf;
    float padding{};
};

struct ShadowRayData {
    PackedVec3f origin;
    float min_distance;
    PackedVec3f direction;
    float max_distance;
    PackedVec3f light_radiance;
    float light_pdf;
};

struct GatherRayData {
    Vec3f radiance;
    Vec2f pixel;
    Vec2f padding{};
};

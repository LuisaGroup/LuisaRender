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
};

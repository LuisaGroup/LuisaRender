//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "compatibility.h"

struct LightData {  // for now, only point light is supported
    Vec3f position;
    Vec3f emission;
};

struct LightSample {
    PackedVec3f radiance;
    float pdf;
    PackedVec3f direction;
    float distance;
};
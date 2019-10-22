//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "compatibility.h"

struct IntersectionData {
    float distance;
    uint triangle_index;
    Vec2f barycentric;
};

struct ShadowIntersectionData {
    float distance;
};

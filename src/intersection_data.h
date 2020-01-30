//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct Intersection {
    float distance;
    uint triangle_index;
    float2 barycentric;
};

struct ShadowIntersection {
    float distance;
};

struct IntersectionData {
    float distance;
    uint triangle_index;
    float2 barycentric;
};

struct ShadowIntersectionData {
    float distance;
};

}

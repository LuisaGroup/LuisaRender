//
// Created by Mike Smith on 2020/2/10.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct ClosestHit {
    float distance;
    uint triangle_index;
    uint instance_index;
    float bary_u;
    float bary_v;
};

struct AnyHit {
    float distance;
};

}

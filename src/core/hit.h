//
// Created by Mike Smith on 2020/2/10.
//

#pragma once

#include "data_types.h"

namespace luisa {

struct ClosestHit {
    float distance;
    uint triangle_index;
    uint instance_index;
    float2 coordinates;
};

struct AnyHit {
    float distance;
};

}

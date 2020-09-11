//
// Created by Mike Smith on 2020/9/11.
//

#pragma once

#include <compute/dsl.h>

namespace luisa::compute {

struct ClosestHit {
    float distance;
    uint triangle_id;
    uint instance_id;
    float bary_u;
    float bary_v;
};

struct AnyHit {
    float distance;
};

}

LUISA_STRUCT(luisa::compute::ClosestHit, distance, triangle_id, instance_id, bary_u, bary_v)
LUISA_STRUCT(luisa::compute::AnyHit, distance)

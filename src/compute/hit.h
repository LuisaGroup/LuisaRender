//
// Created by Mike Smith on 2020/9/11.
//

#pragma once

#include <compute/dsl.h>

namespace luisa::compute {

struct alignas(8) ClosestHit {
    float distance;
    uint triangle_id;
    uint instance_id;
    float2 bary;
};

struct AnyHit {
    float distance;
};

}

LUISA_STRUCT(luisa::compute::ClosestHit, distance, triangle_id, instance_id, bary)
LUISA_STRUCT(luisa::compute::AnyHit, distance)

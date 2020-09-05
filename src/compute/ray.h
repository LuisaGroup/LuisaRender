//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/type_desc.h>

namespace luisa::compute {

struct Ray {
    packed_float3 origin;
    float min_distance;
    packed_float3 direction;
    float max_distance;
};

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

LUISA_STRUCT(luisa::compute::Ray, origin, min_distance, direction, max_distance)
LUISA_STRUCT(luisa::compute::ClosestHit, distance, triangle_id, instance_id, bary_u, bary_v)
LUISA_STRUCT(luisa::compute::AnyHit, distance)

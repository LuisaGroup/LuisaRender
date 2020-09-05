//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/type_desc.h>

namespace luisa::render {

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

LUISA_STRUCT(luisa::render::ClosestHit, distance, triangle_id, instance_id, bary_u, bary_v)
LUISA_STRUCT(luisa::render::AnyHit, distance)

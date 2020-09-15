//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/dsl.h>

namespace luisa::compute {

struct alignas(16) Ray {
    float origin_x;
    float origin_y;
    float origin_z;
    float min_distance;
    float direction_x;
    float direction_y;
    float direction_z;
    float max_distance;
};

}

LUISA_STRUCT(luisa::compute::Ray, origin_x, origin_y, origin_z, min_distance, direction_x, direction_y, direction_z, max_distance)

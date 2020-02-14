//
// Created by Mike Smith on 2020/1/31.
//

#pragma once

#include "data_types.h"

namespace luisa {

struct Ray {
    packed_float3 origin;
    float min_distance;
    packed_float3 direction;
    float max_distance;
};

LUISA_DEVICE_CALLABLE inline auto make_ray(float3 o, float3 d, float t_min = 1e-4f, float t_max = INFINITY) noexcept {
    return Ray{make_packed_float3(o), t_min, make_packed_float3(d), t_max};
}

}

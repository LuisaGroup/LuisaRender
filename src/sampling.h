//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include <core/data_types.h>
#include <core/mathematics.h>

namespace luisa {

LUISA_DEVICE_CALLABLE inline float3 cosine_sample_hemisphere(float u1, float u2) {
    using namespace math;
    auto r = sqrt(u1);
    auto phi = 2.0f * PI * u2;
    auto x = r * cos(phi);
    auto y = r * sin(phi);
    return make_float3(x, y, sqrt(max(0.0f, 1.0f - x * x - y * y)));
}

LUISA_DEVICE_CALLABLE inline float2 concentric_sample_disk(float r1, float r2) {
    
    auto offset = 2.0f * make_float2(r1, r2) - make_float2(1.0f, 1.0f);
    
    if (offset.x == 0 && offset.y == 0) { return make_float2(); }
    
    auto theta = 0.0f;
    auto r = 0.0f;
    
    using namespace math;
    if (abs(offset.x) > abs(offset.y)) {
        r = offset.x;
        theta = PI_OVER_FOUR * (offset.y / offset.x);
    } else {
        r = offset.y;
        theta = PI_OVER_TWO - PI_OVER_FOUR * (offset.x / offset.y);
    }
    return make_float2(r * cos(theta), r * sin(theta));
}

}
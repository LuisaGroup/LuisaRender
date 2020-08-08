//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include <core/data_types.h>
#include <core/mathematics.h>

namespace luisa { inline namespace sampling {

LUISA_DEVICE_CALLABLE inline float3 cosine_sample_hemisphere(float u1, float u2) noexcept {
    auto r = sqrt(u1);
    auto phi = 2.0f * math::PI * u2;
    auto x = static_cast<float>(r * cos(phi));
    auto y = static_cast<float>(r * sin(phi));
    return make_float3(x, y, math::sqrt(math::max(0.0f, 1.0f - x * x - y * y)));
}

LUISA_DEVICE_CALLABLE inline float2 concentric_sample_disk(float r1, float r2) noexcept {
    
    using namespace math;
    
    auto offset = 2.0f * make_float2(r1, r2) - make_float2(1.0f, 1.0f);
    
    if (offset.x == 0 && offset.y == 0) { return make_float2(); }
    
    auto theta = 0.0f;
    auto r = 0.0f;
    
    if (abs(offset.x) > abs(offset.y)) {
        r = offset.x;
        theta = math::PI_OVER_FOUR * (offset.y / offset.x);
    } else {
        r = offset.y;
        theta = PI_OVER_TWO - PI_OVER_FOUR * (offset.x / offset.y);
    }
    return make_float2(r * cos(theta), r * sin(theta));
}

LUISA_DEVICE_CALLABLE inline float balance_heuristic(float pa, float pb) noexcept {
    using namespace math;
    return pa / max(pa + pb, 1e-4f);
}

LUISA_DEVICE_CALLABLE inline uint sample_discrete(LUISA_DEVICE_SPACE const float *cdf, uint begin, uint end, float value) noexcept {
    
    auto count = end - begin;
    auto p = begin;
    
    while (count > 0) {
        auto step = count / 2;
        auto mid = p + step;
        if (cdf[mid] < value) {
            p = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return math::clamp(p, begin, end - 1u);
}

LUISA_DEVICE_CALLABLE inline float2 uniform_sample_triangle(float u0, float u1) noexcept {
    auto su0 = sqrt(u0);
    auto b0 = 1 - su0;
    auto b1 = u1 * su0;
    return make_float2(b0, b1);
}

}}

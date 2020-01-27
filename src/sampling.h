//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "compatibility.h"

inline Vec3f cosine_sample_hemisphere(float u1, float u2) {
    using namespace metal;
    auto r = sqrt(u1);
    auto phi = 2.0f * M_PIf * u2;
    auto x = r * cos(phi);
    auto y = r * sin(phi);
    return {x, y, sqrt(max(0.0f, 1.0f - x * x - y * y))};
}

inline Vec2f concentric_sample_disk(float r1, float r2) {
    
    auto offset = 2.0f * Vec2f{r1, r2} - Vec2f{1.0f, 1.0f};
    
    if (offset.x == 0 && offset.y == 0) { return {}; }
    
    auto theta = 0.0f;
    auto r = 0.0f;
    
    using namespace metal;
    if (abs(offset.x) > abs(offset.y)) {
        r = offset.x;
        theta = M_PI_4f * (offset.y / offset.x);
    } else {
        r = offset.y;
        theta = M_PI_2f - M_PI_4f * (offset.x / offset.y);
    }
    return {r * cos(theta), r * sin(theta)};
}
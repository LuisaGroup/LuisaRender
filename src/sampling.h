//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "compatibility.h"

Vec3f cosine_sample_hemisphere(const float u1, const float u2) {
    using namespace metal;
    auto r = sqrt(u1);
    auto phi = 2.0f * M_PIf * u2;
    auto x = r * cos(phi);
    auto y = r * sin(phi);
    return {x, y, sqrt(max(0.0f, 1.0f - x * x - y * y))};
}

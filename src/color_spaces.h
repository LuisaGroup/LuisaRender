//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "compatibility.h"

inline float luminance_rgb(Vec3f rgb) {
    return rgb.x * 0.212671f + rgb.y * 0.715160f + rgb.z * 0.072169f;
}

inline float luminance_xyz(Vec3f xyz) {
    return xyz.y;
}

inline Vec3f xyz2rgb(Vec3f xyz) {
    auto r = 3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z;
    auto g = -0.969256f * xyz.x + 1.875991f * xyz.y + 0.041556f * xyz.z;
    auto b = 0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z;
    return {r, g, b};
}

inline Vec3f rgb2xyz(const Vec3f rgb) {
    auto x = 0.412453f * rgb.r + 0.357580f * rgb.g + 0.180423f * rgb.b;
    auto y = 0.212671f * rgb.r + 0.715160f * rgb.g + 0.072169f * rgb.b;
    auto z = 0.019334f * rgb.r + 0.119193f * rgb.g + 0.950227f * rgb.b;
    return {x, y, z};
}

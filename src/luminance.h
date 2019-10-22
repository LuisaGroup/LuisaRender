//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "compatibility.h"

inline float luminance(Vec3f rgb) {
    return rgb.x * 0.212671f + rgb.y * 0.715160f + rgb.z * 0.072169f;
}

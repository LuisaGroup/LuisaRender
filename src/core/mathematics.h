//
// Created by Mike Smith on 2020/1/30.
//

#pragma once

#include "data_types.h"

namespace luisa::math { inline namespace constants {

LUISA_CONSTANT_SPACE float PI[[maybe_unused]] = 3.14159265358979323846264338327950288f;
LUISA_CONSTANT_SPACE float PI_OVER_TWO[[maybe_unused]] = 1.57079632679489661923132169163975144f;
LUISA_CONSTANT_SPACE float PI_OVER_FOUR[[maybe_unused]] = 0.785398163397448309615660845819875721f;
LUISA_CONSTANT_SPACE float INV_PI[[maybe_unused]] = 0.318309886183790671537767526745028724f;
LUISA_CONSTANT_SPACE float TWO_OVER_PI[[maybe_unused]] = 0.636619772367581343075535053490057448f;
LUISA_CONSTANT_SPACE float SQRT_TWO[[maybe_unused]] = 1.41421356237309504880168872420969808f;
LUISA_CONSTANT_SPACE float INV_SQRT_TWO[[maybe_unused]] = 0.707106781186547524400844362104849039f;

}}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>

namespace luisa::math {

// scalar functions

using glm::cos;
using glm::sin;
using glm::tan;
using glm::acos;
using glm::asin;
using glm::atan;
using std::atan2;

using glm::ceil;
using glm::floor;
using glm::round;

using glm::pow;
using glm::exp;
using glm::log;
using glm::log2;
using std::log10;

using glm::min;
using glm::max;

using glm::abs;
using glm::clamp;

inline float radians(float deg) noexcept { return deg * constants::PI / 180.0f; }
inline float degrees(float rad) noexcept { return rad * constants::INV_PI * 180.0f; }

// vector functions

using glm::normalize;
using glm::length;
using glm::dot;
using glm::cross;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_COS
using glm::cos;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_SIN
using glm::sin;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_TAN
using glm::tan;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_ACOS
using glm::acos;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_ASIN
using glm::asin;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN
using glm::atan;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_CEIL
using glm::ceil;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_FLOOR
using glm::floor;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_ROUND
using glm::round;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_LOG
using glm::log;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_EXP
using glm::exp;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_LOG2
using glm::log2;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_POW
using glm::pow;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_MIN
using glm::min;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_MAX
using glm::max;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_ABS
using glm::abs;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_CLAMP
using glm::clamp;

// matrix functions

}

#endif

namespace luisa::math {

LUISA_DEVICE_CALLABLE inline float max_component(float2 v) noexcept { return max(v.x, v.y); }
LUISA_DEVICE_CALLABLE inline float min_component(float2 v) noexcept { return min(v.x, v.y); }
LUISA_DEVICE_CALLABLE inline float max_component(float3 v) noexcept { return max(max(v.x, v.y), v.z); }
LUISA_DEVICE_CALLABLE inline float min_component(float3 v) noexcept { return min(min(v.x, v.y), v.z); }
LUISA_DEVICE_CALLABLE inline float max_component(float4 v) noexcept { return max(max(v.x, v.y), max(v.z, v.w)); }
LUISA_DEVICE_CALLABLE inline float min_component(float4 v) noexcept { return min(min(v.x, v.y), min(v.z, v.w)); }

LUISA_DEVICE_CALLABLE inline int max_component(int2 v) noexcept { return max(v.x, v.y); }
LUISA_DEVICE_CALLABLE inline int min_component(int2 v) noexcept { return min(v.x, v.y); }
LUISA_DEVICE_CALLABLE inline int max_component(int3 v) noexcept { return max(max(v.x, v.y), v.z); }
LUISA_DEVICE_CALLABLE inline int min_component(int3 v) noexcept { return min(min(v.x, v.y), v.z); }
LUISA_DEVICE_CALLABLE inline int max_component(int4 v) noexcept { return max(max(v.x, v.y), max(v.z, v.w)); }
LUISA_DEVICE_CALLABLE inline int min_component(int4 v) noexcept { return min(min(v.x, v.y), min(v.z, v.w)); }

LUISA_DEVICE_CALLABLE inline uint max_component(uint2 v) noexcept { return max(v.x, v.y); }
LUISA_DEVICE_CALLABLE inline uint min_component(uint2 v) noexcept { return min(v.x, v.y); }
LUISA_DEVICE_CALLABLE inline uint max_component(uint3 v) noexcept { return max(max(v.x, v.y), v.z); }
LUISA_DEVICE_CALLABLE inline uint min_component(uint3 v) noexcept { return min(min(v.x, v.y), v.z); }
LUISA_DEVICE_CALLABLE inline uint max_component(uint4 v) noexcept { return max(max(v.x, v.y), max(v.z, v.w)); }
LUISA_DEVICE_CALLABLE inline uint min_component(uint4 v) noexcept { return min(min(v.x, v.y), min(v.z, v.w)); }

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_COS
LUISA_DEVICE_CALLABLE inline float2 cos(float2 v) noexcept { return make_float2(luisa::math::cos(v.x), luisa::math::cos(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 cos(float3 v) noexcept { return make_float3(luisa::math::cos(v.x), luisa::math::cos(v.y), luisa::math::cos(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 cos(float4 v) noexcept { return make_float4(luisa::math::cos(v.x), luisa::math::cos(v.y), luisa::math::cos(v.z), luisa::math::cos(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_SIN
LUISA_DEVICE_CALLABLE inline float2 sin(float2 v) noexcept { return make_float2(luisa::math::sin(v.x), luisa::math::sin(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 sin(float3 v) noexcept { return make_float3(luisa::math::sin(v.x), luisa::math::sin(v.y), luisa::math::sin(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 sin(float4 v) noexcept { return make_float4(luisa::math::sin(v.x), luisa::math::sin(v.y), luisa::math::sin(v.z), luisa::math::sin(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_TAN
LUISA_DEVICE_CALLABLE inline float2 tan(float2 v) noexcept { return make_float2(luisa::math::tan(v.x), luisa::math::tan(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 tan(float3 v) noexcept { return make_float3(luisa::math::tan(v.x), luisa::math::tan(v.y), luisa::math::tan(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 tan(float4 v) noexcept { return make_float4(luisa::math::tan(v.x), luisa::math::tan(v.y), luisa::math::tan(v.z), luisa::math::tan(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ACOS
LUISA_DEVICE_CALLABLE inline float2 acos(float2 v) noexcept { return make_float2(luisa::math::acos(v.x), luisa::math::acos(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 acos(float3 v) noexcept { return make_float3(luisa::math::acos(v.x), luisa::math::acos(v.y), luisa::math::acos(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 acos(float4 v) noexcept { return make_float4(luisa::math::acos(v.x), luisa::math::acos(v.y), luisa::math::acos(v.z), luisa::math::acos(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ASIN
LUISA_DEVICE_CALLABLE inline float2 asin(float2 v) noexcept { return make_float2(luisa::math::asin(v.x), luisa::math::asin(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 asin(float3 v) noexcept { return make_float3(luisa::math::asin(v.x), luisa::math::asin(v.y), luisa::math::asin(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 asin(float4 v) noexcept { return make_float4(luisa::math::asin(v.x), luisa::math::asin(v.y), luisa::math::asin(v.z), luisa::math::asin(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN
LUISA_DEVICE_CALLABLE inline float2 atan(float2 v) noexcept { return make_float2(luisa::math::atan(v.x), luisa::math::atan(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 atan(float3 v) noexcept { return make_float3(luisa::math::atan(v.x), luisa::math::atan(v.y), luisa::math::atan(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 atan(float4 v) noexcept { return make_float4(luisa::math::atan(v.x), luisa::math::atan(v.y), luisa::math::atan(v.z), luisa::math::atan(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN2
LUISA_DEVICE_CALLABLE inline float2 atan2(float2 y, float2 x) noexcept { return make_float2(luisa::math::atan2(y.x, x.x), luisa::math::atan2(y.y, x.y)); }
LUISA_DEVICE_CALLABLE inline float3 atan2(float3 y, float3 x) noexcept {
    return make_float3(luisa::math::atan2(y.x, x.x),
                       luisa::math::atan2(y.y, x.y),
                       luisa::math::atan2(y.z, x.z));
}
LUISA_DEVICE_CALLABLE inline float4 atan2(float4 y, float4 x) noexcept {
    return make_float4(luisa::math::atan2(y.x, x.x),
                       luisa::math::atan2(y.y, x.y),
                       luisa::math::atan2(y.z, x.z),
                       luisa::math::atan2(y.w, x.w));
}
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_CEIL
LUISA_DEVICE_CALLABLE inline float2 ceil(float2 v) noexcept { return make_float2(luisa::math::ceil(v.x), luisa::math::ceil(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 ceil(float3 v) noexcept { return make_float3(luisa::math::ceil(v.x), luisa::math::ceil(v.y), luisa::math::ceil(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 ceil(float4 v) noexcept { return make_float4(luisa::math::ceil(v.x), luisa::math::ceil(v.y), luisa::math::ceil(v.z), luisa::math::ceil(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_FLOOR
LUISA_DEVICE_CALLABLE inline float2 floor(float2 v) noexcept { return make_float2(luisa::math::floor(v.x), luisa::math::floor(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 floor(float3 v) noexcept { return make_float3(luisa::math::floor(v.x), luisa::math::floor(v.y), luisa::math::floor(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 floor(float4 v) noexcept { return make_float4(luisa::math::floor(v.x), luisa::math::floor(v.y), luisa::math::floor(v.z), luisa::math::floor(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ROUND
LUISA_DEVICE_CALLABLE inline float2 round(float2 v) noexcept { return make_float2(luisa::math::round(v.x), luisa::math::round(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 round(float3 v) noexcept { return make_float3(luisa::math::round(v.x), luisa::math::round(v.y), luisa::math::round(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 round(float4 v) noexcept { return make_float4(luisa::math::round(v.x), luisa::math::round(v.y), luisa::math::round(v.z), luisa::math::round(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_LOG
LUISA_DEVICE_CALLABLE inline float2 log(float2 v) noexcept { return make_float2(luisa::math::log(v.x), luisa::math::log(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 log(float3 v) noexcept { return make_float3(luisa::math::log(v.x), luisa::math::log(v.y), luisa::math::log(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 log(float4 v) noexcept { return make_float4(luisa::math::log(v.x), luisa::math::log(v.y), luisa::math::log(v.z), luisa::math::log(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_EXP
LUISA_DEVICE_CALLABLE inline float2 exp(float2 v) noexcept { return make_float2(luisa::math::exp(v.x), luisa::math::exp(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 exp(float3 v) noexcept { return make_float3(luisa::math::exp(v.x), luisa::math::exp(v.y), luisa::math::exp(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 exp(float4 v) noexcept { return make_float4(luisa::math::exp(v.x), luisa::math::exp(v.y), luisa::math::exp(v.z), luisa::math::exp(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_LOG2
LUISA_DEVICE_CALLABLE inline float2 log2(float2 v) noexcept { return make_float2(luisa::math::log2(v.x), luisa::math::log2(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 log2(float3 v) noexcept { return make_float3(luisa::math::log2(v.x), luisa::math::log2(v.y), luisa::math::log2(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 log2(float4 v) noexcept { return make_float4(luisa::math::log2(v.x), luisa::math::log2(v.y), luisa::math::log2(v.z), luisa::math::log2(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_LOG10
LUISA_DEVICE_CALLABLE inline float2 log10(float2 v) noexcept { return make_float2(luisa::math::log10(v.x), luisa::math::log10(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 log10(float3 v) noexcept { return make_float3(luisa::math::log10(v.x), luisa::math::log10(v.y), luisa::math::log10(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 log10(float4 v) noexcept {
    return make_float4(luisa::math::log10(v.x),
                       luisa::math::log10(v.y),
                       luisa::math::log10(v.z),
                       luisa::math::log10(v.w));
}
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_POW
LUISA_DEVICE_CALLABLE inline float2 pow(float2 a, float2 b) noexcept { return make_float2(luisa::math::pow(a.x, b.x), luisa::math::pow(a.y, b.y)); }
LUISA_DEVICE_CALLABLE inline float3 pow(float3 a, float3 b) noexcept { return make_float3(luisa::math::pow(a.x, b.x), luisa::math::pow(a.y, b.y), luisa::math::pow(a.z, b.z)); }
LUISA_DEVICE_CALLABLE inline float4 pow(float4 a, float4 b) noexcept { return make_float4(luisa::math::pow(a.x, b.x), luisa::math::pow(a.y, b.y), luisa::math::pow(a.z, b.z), luisa::math::pow(a.w, b.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_MIN
LUISA_DEVICE_CALLABLE inline float2 min(float2 a, float2 b) noexcept { return make_float2(luisa::math::min(a.x, b.x), luisa::math::min(a.y, b.y)); }
LUISA_DEVICE_CALLABLE inline float3 min(float3 a, float3 b) noexcept { return make_float3(luisa::math::min(a.x, b.x), luisa::math::min(a.y, b.y), luisa::math::min(a.z, b.z)); }
LUISA_DEVICE_CALLABLE inline float4 min(float4 a, float4 b) noexcept { return make_float4(luisa::math::min(a.x, b.x), luisa::math::min(a.y, b.y), luisa::math::min(a.z, b.z), luisa::math::min(a.w, b.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_MAX
LUISA_DEVICE_CALLABLE inline float2 max(float2 a, float2 b) noexcept { return make_float2(luisa::math::max(a.x, b.x), luisa::math::max(a.y, b.y)); }
LUISA_DEVICE_CALLABLE inline float3 max(float3 a, float3 b) noexcept { return make_float3(luisa::math::max(a.x, b.x), luisa::math::max(a.y, b.y), luisa::math::max(a.z, b.z)); }
LUISA_DEVICE_CALLABLE inline float4 max(float4 a, float4 b) noexcept { return make_float4(luisa::math::max(a.x, b.x), luisa::math::max(a.y, b.y), luisa::math::max(a.z, b.z), luisa::math::max(a.w, b.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ABS
LUISA_DEVICE_CALLABLE inline float2 abs(float2 v) noexcept { return make_float2(luisa::math::abs(v.x), luisa::math::abs(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 abs(float3 v) noexcept { return make_float3(luisa::math::abs(v.x), luisa::math::abs(v.y), luisa::math::abs(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 abs(float4 v) noexcept { return make_float4(luisa::math::abs(v.x), luisa::math::abs(v.y), luisa::math::abs(v.z), luisa::math::abs(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_CLAMP
LUISA_DEVICE_CALLABLE inline float2 clamp(float2 v, float lo, float hi) noexcept { return make_float2(luisa::math::clamp(v.x, lo, hi), luisa::math::clamp(v.y, lo, hi)); }
LUISA_DEVICE_CALLABLE inline float3 clamp(float3 v, float lo, float hi) noexcept { return make_float3(luisa::math::clamp(v.x, lo, hi), luisa::math::clamp(v.y, lo, hi), luisa::math::clamp(v.z, lo, hi)); }
LUISA_DEVICE_CALLABLE inline float4 clamp(float4 v, float lo, float hi) noexcept {
    return make_float4(luisa::math::clamp(v.x, lo, hi),
                       luisa::math::clamp(v.y, lo, hi),
                       luisa::math::clamp(v.z, lo, hi),
                       luisa::math::clamp(v.w, lo, hi));
}
LUISA_DEVICE_CALLABLE inline float2 clamp(float2 v, float2 lo, float2 hi) noexcept { return make_float2(luisa::math::clamp(v.x, lo.x, hi.x), luisa::math::clamp(v.y, lo.y, hi.y)); }
LUISA_DEVICE_CALLABLE inline float3 clamp(float3 v, float3 lo, float3 hi) noexcept { return make_float3(luisa::math::clamp(v.x, lo.x, hi.x), luisa::math::clamp(v.y, lo.y, hi.y), luisa::math::clamp(v.z, lo.z, hi.z)); }
LUISA_DEVICE_CALLABLE inline float4 clamp(float4 v, float4 lo, float4 hi) noexcept {
    return make_float4(luisa::math::clamp(v.x, lo.x, hi.x),
                       luisa::math::clamp(v.y, lo.y, hi.y),
                       luisa::math::clamp(v.z, lo.z, hi.z),
                       luisa::math::clamp(v.w, lo.w, hi.w));
}
#endif

}

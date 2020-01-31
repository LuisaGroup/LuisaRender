//
// Created by Mike Smith on 2020/1/30.
//

#pragma once

#include "data_types.h"

namespace luisa::math { inline namespace constants {

LUISA_CONSTANT_SPACE constexpr auto PI[[maybe_unused]] = 3.14159265358979323846264338327950288f;
LUISA_CONSTANT_SPACE constexpr auto PI_OVER_TWO[[maybe_unused]] = 1.57079632679489661923132169163975144f;
LUISA_CONSTANT_SPACE constexpr auto PI_OVER_FOUR[[maybe_unused]] = 0.785398163397448309615660845819875721f;
LUISA_CONSTANT_SPACE constexpr auto INV_PI[[maybe_unused]] = 0.318309886183790671537767526745028724f;
LUISA_CONSTANT_SPACE constexpr auto TWO_OVER_PI[[maybe_unused]] = 0.636619772367581343075535053490057448f;
LUISA_CONSTANT_SPACE constexpr auto SQRT_TWO[[maybe_unused]] = 1.41421356237309504880168872420969808f;
LUISA_CONSTANT_SPACE constexpr auto INV_SQRT_TWO[[maybe_unused]] = 0.707106781186547524400844362104849039f;
    
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
using glm::distance;

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
    
}

#endif

namespace luisa::math {

LUISA_DEVICE_CALLABLE inline bool all_zero(float2 v) noexcept { return v.x == 0 && v.y == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(float3 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(packed_float3 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(float4 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0 && v.w == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(uint2 v) noexcept { return v.x == 0 && v.y == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(uint3 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(packed_uint3 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(uint4 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0 && v.w == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(int2 v) noexcept { return v.x == 0 && v.y == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(int3 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(packed_int3 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0; }
LUISA_DEVICE_CALLABLE inline bool all_zero(int4 v) noexcept { return v.x == 0 && v.y == 0 && v.z == 0 && v.w == 0; }

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
LUISA_DEVICE_CALLABLE inline float2 cos(float2 v) noexcept { return make_float2(math::cos(v.x), math::cos(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 cos(float3 v) noexcept { return make_float3(math::cos(v.x), math::cos(v.y), math::cos(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 cos(float4 v) noexcept { return make_float4(math::cos(v.x), math::cos(v.y), math::cos(v.z), math::cos(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_SIN
LUISA_DEVICE_CALLABLE inline float2 sin(float2 v) noexcept { return make_float2(math::sin(v.x), math::sin(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 sin(float3 v) noexcept { return make_float3(math::sin(v.x), math::sin(v.y), math::sin(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 sin(float4 v) noexcept { return make_float4(math::sin(v.x), math::sin(v.y), math::sin(v.z), math::sin(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_TAN
LUISA_DEVICE_CALLABLE inline float2 tan(float2 v) noexcept { return make_float2(math::tan(v.x), math::tan(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 tan(float3 v) noexcept { return make_float3(math::tan(v.x), math::tan(v.y), math::tan(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 tan(float4 v) noexcept { return make_float4(math::tan(v.x), math::tan(v.y), math::tan(v.z), math::tan(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ACOS
LUISA_DEVICE_CALLABLE inline float2 acos(float2 v) noexcept { return make_float2(math::acos(v.x), math::acos(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 acos(float3 v) noexcept { return make_float3(math::acos(v.x), math::acos(v.y), math::acos(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 acos(float4 v) noexcept { return make_float4(math::acos(v.x), math::acos(v.y), math::acos(v.z), math::acos(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ASIN
LUISA_DEVICE_CALLABLE inline float2 asin(float2 v) noexcept { return make_float2(math::asin(v.x), math::asin(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 asin(float3 v) noexcept { return make_float3(math::asin(v.x), math::asin(v.y), math::asin(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 asin(float4 v) noexcept { return make_float4(math::asin(v.x), math::asin(v.y), math::asin(v.z), math::asin(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN
LUISA_DEVICE_CALLABLE inline float2 atan(float2 v) noexcept { return make_float2(math::atan(v.x), math::atan(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 atan(float3 v) noexcept { return make_float3(math::atan(v.x), math::atan(v.y), math::atan(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 atan(float4 v) noexcept { return make_float4(math::atan(v.x), math::atan(v.y), math::atan(v.z), math::atan(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN2
LUISA_DEVICE_CALLABLE inline float2 atan2(float2 y, float2 x) noexcept { return make_float2(math::atan2(y.x, x.x), math::atan2(y.y, x.y)); }
LUISA_DEVICE_CALLABLE inline float3 atan2(float3 y, float3 x) noexcept {
    return make_float3(math::atan2(y.x, x.x),
                       math::atan2(y.y, x.y),
                       math::atan2(y.z, x.z));
}
LUISA_DEVICE_CALLABLE inline float4 atan2(float4 y, float4 x) noexcept {
    return make_float4(math::atan2(y.x, x.x),
                       math::atan2(y.y, x.y),
                       math::atan2(y.z, x.z),
                       math::atan2(y.w, x.w));
}
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_CEIL
LUISA_DEVICE_CALLABLE inline float2 ceil(float2 v) noexcept { return make_float2(math::ceil(v.x), math::ceil(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 ceil(float3 v) noexcept { return make_float3(math::ceil(v.x), math::ceil(v.y), math::ceil(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 ceil(float4 v) noexcept { return make_float4(math::ceil(v.x), math::ceil(v.y), math::ceil(v.z), math::ceil(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_FLOOR
LUISA_DEVICE_CALLABLE inline float2 floor(float2 v) noexcept { return make_float2(math::floor(v.x), math::floor(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 floor(float3 v) noexcept { return make_float3(math::floor(v.x), math::floor(v.y), math::floor(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 floor(float4 v) noexcept { return make_float4(math::floor(v.x), math::floor(v.y), math::floor(v.z), math::floor(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ROUND
LUISA_DEVICE_CALLABLE inline float2 round(float2 v) noexcept { return make_float2(math::round(v.x), math::round(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 round(float3 v) noexcept { return make_float3(math::round(v.x), math::round(v.y), math::round(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 round(float4 v) noexcept { return make_float4(math::round(v.x), math::round(v.y), math::round(v.z), math::round(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_LOG
LUISA_DEVICE_CALLABLE inline float2 log(float2 v) noexcept { return make_float2(math::log(v.x), math::log(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 log(float3 v) noexcept { return make_float3(math::log(v.x), math::log(v.y), math::log(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 log(float4 v) noexcept { return make_float4(math::log(v.x), math::log(v.y), math::log(v.z), math::log(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_EXP
LUISA_DEVICE_CALLABLE inline float2 exp(float2 v) noexcept { return make_float2(math::exp(v.x), math::exp(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 exp(float3 v) noexcept { return make_float3(math::exp(v.x), math::exp(v.y), math::exp(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 exp(float4 v) noexcept { return make_float4(math::exp(v.x), math::exp(v.y), math::exp(v.z), math::exp(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_LOG2
LUISA_DEVICE_CALLABLE inline float2 log2(float2 v) noexcept { return make_float2(math::log2(v.x), math::log2(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 log2(float3 v) noexcept { return make_float3(math::log2(v.x), math::log2(v.y), math::log2(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 log2(float4 v) noexcept { return make_float4(math::log2(v.x), math::log2(v.y), math::log2(v.z), math::log2(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_LOG10
LUISA_DEVICE_CALLABLE inline float2 log10(float2 v) noexcept { return make_float2(math::log10(v.x), math::log10(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 log10(float3 v) noexcept { return make_float3(math::log10(v.x), math::log10(v.y), math::log10(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 log10(float4 v) noexcept {
    return make_float4(math::log10(v.x),
                       math::log10(v.y),
                       math::log10(v.z),
                       math::log10(v.w));
}
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_POW
LUISA_DEVICE_CALLABLE inline float2 pow(float2 a, float2 b) noexcept { return make_float2(math::pow(a.x, b.x), math::pow(a.y, b.y)); }
LUISA_DEVICE_CALLABLE inline float3 pow(float3 a, float3 b) noexcept { return make_float3(math::pow(a.x, b.x), math::pow(a.y, b.y), math::pow(a.z, b.z)); }
LUISA_DEVICE_CALLABLE inline float4 pow(float4 a, float4 b) noexcept { return make_float4(math::pow(a.x, b.x), math::pow(a.y, b.y), math::pow(a.z, b.z), math::pow(a.w, b.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_MIN
LUISA_DEVICE_CALLABLE inline float2 min(float2 a, float2 b) noexcept { return make_float2(math::min(a.x, b.x), math::min(a.y, b.y)); }
LUISA_DEVICE_CALLABLE inline float3 min(float3 a, float3 b) noexcept { return make_float3(math::min(a.x, b.x), math::min(a.y, b.y), math::min(a.z, b.z)); }
LUISA_DEVICE_CALLABLE inline float4 min(float4 a, float4 b) noexcept { return make_float4(math::min(a.x, b.x), math::min(a.y, b.y), math::min(a.z, b.z), math::min(a.w, b.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_MAX
LUISA_DEVICE_CALLABLE inline float2 max(float2 a, float2 b) noexcept { return make_float2(math::max(a.x, b.x), math::max(a.y, b.y)); }
LUISA_DEVICE_CALLABLE inline float3 max(float3 a, float3 b) noexcept { return make_float3(math::max(a.x, b.x), math::max(a.y, b.y), math::max(a.z, b.z)); }
LUISA_DEVICE_CALLABLE inline float4 max(float4 a, float4 b) noexcept { return make_float4(math::max(a.x, b.x), math::max(a.y, b.y), math::max(a.z, b.z), math::max(a.w, b.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_ABS
LUISA_DEVICE_CALLABLE inline float2 abs(float2 v) noexcept { return make_float2(math::abs(v.x), math::abs(v.y)); }
LUISA_DEVICE_CALLABLE inline float3 abs(float3 v) noexcept { return make_float3(math::abs(v.x), math::abs(v.y), math::abs(v.z)); }
LUISA_DEVICE_CALLABLE inline float4 abs(float4 v) noexcept { return make_float4(math::abs(v.x), math::abs(v.y), math::abs(v.z), math::abs(v.w)); }
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_VECTOR_CLAMP
LUISA_DEVICE_CALLABLE inline float2 clamp(float2 v, float lo, float hi) noexcept { return make_float2(math::clamp(v.x, lo, hi), math::clamp(v.y, lo, hi)); }
LUISA_DEVICE_CALLABLE inline float3 clamp(float3 v, float lo, float hi) noexcept { return make_float3(math::clamp(v.x, lo, hi), math::clamp(v.y, lo, hi), math::clamp(v.z, lo, hi)); }
LUISA_DEVICE_CALLABLE inline float4 clamp(float4 v, float lo, float hi) noexcept {
    return make_float4(math::clamp(v.x, lo, hi),
                       math::clamp(v.y, lo, hi),
                       math::clamp(v.z, lo, hi),
                       math::clamp(v.w, lo, hi));
}
LUISA_DEVICE_CALLABLE inline float2 clamp(float2 v, float2 lo, float2 hi) noexcept { return make_float2(math::clamp(v.x, lo.x, hi.x), math::clamp(v.y, lo.y, hi.y)); }
LUISA_DEVICE_CALLABLE inline float3 clamp(float3 v, float3 lo, float3 hi) noexcept { return make_float3(math::clamp(v.x, lo.x, hi.x), math::clamp(v.y, lo.y, hi.y), math::clamp(v.z, lo.z, hi.z)); }
LUISA_DEVICE_CALLABLE inline float4 clamp(float4 v, float4 lo, float4 hi) noexcept {
    return make_float4(math::clamp(v.x, lo.x, hi.x),
                       math::clamp(v.y, lo.y, hi.y),
                       math::clamp(v.z, lo.z, hi.z),
                       math::clamp(v.w, lo.w, hi.w));
}
#endif

#ifndef LUISA_MATH_HAS_BUILTIN_MATRIX_TRANSPOSE

LUISA_DEVICE_CALLABLE inline auto transpose(float3x3 m) noexcept {
    return make_float3x3(
        m[0].x, m[1].x, m[2].x,
        m[0].y, m[1].y, m[2].y,
        m[0].z, m[1].z, m[2].z);
}

LUISA_DEVICE_CALLABLE inline auto transpose(float4x4 m) noexcept {
    return make_float4x4(
        m[0].x, m[1].x, m[2].x, m[3].x,
        m[0].y, m[1].y, m[2].y, m[3].y,
        m[0].z, m[1].z, m[2].z, m[3].z,
        m[0].w, m[1].w, m[2].w, m[3].w);
}

#endif

#ifndef LUISA_MATH_HAS_BUILTIN_MATRIX_INVERSE

LUISA_DEVICE_CALLABLE inline auto inverse(float3x3 m) noexcept {  // from GLM
    
    auto one_over_determinant = 1.0f / (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
                                        m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
                                        m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
    
    return make_float3x3(
        (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
        (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
        (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
        (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
        (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
        (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
        (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
        (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
        (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
}

LUISA_DEVICE_CALLABLE inline auto inverse(float4x4 m) noexcept {  // from GLM
    
    auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    
    auto fac0 = make_float4(coef00, coef00, coef02, coef03);
    auto fac1 = make_float4(coef04, coef04, coef06, coef07);
    auto fac2 = make_float4(coef08, coef08, coef10, coef11);
    auto fac3 = make_float4(coef12, coef12, coef14, coef15);
    auto fac4 = make_float4(coef16, coef16, coef18, coef19);
    auto fac5 = make_float4(coef20, coef20, coef22, coef23);
    
    auto Vec0 = make_float4(m[1].x, m[0].x, m[0].x, m[0].x);
    auto Vec1 = make_float4(m[1].y, m[0].y, m[0].y, m[0].y);
    auto Vec2 = make_float4(m[1].z, m[0].z, m[0].z, m[0].z);
    auto Vec3 = make_float4(m[1].w, m[0].w, m[0].w, m[0].w);
    
    auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    
    auto sign_a = make_float4(+1, -1, +1, -1);
    auto sign_b = make_float4(-1, +1, -1, +1);
    
    auto inv = make_float4x4(inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b);
    
    auto dot0 = m[0] * make_float4(inv[0].x, inv[1].x, inv[2].x, inv[3].x);
    auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    
    auto one_over_determinant = 1.0f / dot1;
    return inv * one_over_determinant;
}

#endif

LUISA_DEVICE_CALLABLE inline auto identity() noexcept {
    return make_float4x4();
}

LUISA_DEVICE_CALLABLE inline auto translation(float3 v) noexcept {
    return make_float4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        v.x, v.y, v.z, 1.0f);
}

LUISA_DEVICE_CALLABLE inline auto translation(float tx, float ty, float tz) noexcept {
    return translation(make_float3(tx, ty, tz));
}

LUISA_DEVICE_CALLABLE inline auto scaling(float3 s) noexcept {
    return make_float4x4(
        s.x, 0.0f, 0.0f, 0.0f,
        0.0f, s.y, 0.0f, 0.0f,
        0.0f, 0.0f, s.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

LUISA_DEVICE_CALLABLE inline auto scaling(float sx, float sy, float sz) noexcept {
    return scaling(make_float3(sx, sy, sz));
}

LUISA_DEVICE_CALLABLE inline auto scaling(float s) noexcept {
    return scaling(make_float3(s));
}

LUISA_DEVICE_CALLABLE inline auto rotation(float3 axis, float angle) noexcept {
    
    auto c = cos(angle);
    auto s = sin(angle);
    auto a = normalize(axis);
    auto t = (1.0f - c) * a;
    
    return make_float4x4(
        c + t.x * a.x, t.x * a.y + s * a.z, t.x * a.z - s * a.y, 0.0f,
        t.y * a.x - s * a.z, c + t.y * a.y, t.y * a.z + s * a.x, 0.0f,
        t.z * a.x + s * a.y, t.z * a.y - s * a.x, c + t.z * a.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}
    
}

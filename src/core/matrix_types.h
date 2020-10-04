//
// Created by Mike on 8/28/2020.
//

#pragma once

#include "vector_types.h"

namespace luisa {

inline namespace matrix {

struct float3x3 {

    float3 cols[3];

    explicit constexpr float3x3(float s = 1.0f) noexcept
        : cols{make_float3(s, 0.0f, 0.0f), make_float3(0.0f, s, 0.0f), make_float3(0.0f, 0.0f, s)} {}

    constexpr float3x3(const float3 c0, const float3 c1, const float3 c2) noexcept
        : cols{c0, c1, c2} {}

    constexpr float3x3(float m00, float m01, float m02,
                       float m10, float m11, float m12,
                       float m20, float m21, float m22) noexcept
        : cols{make_float3(m00, m01, m02), make_float3(m10, m11, m12), make_float3(m20, m21, m22)} {}

    template<typename Index>
    [[nodiscard]] float3 &operator[](Index i) noexcept { return cols[i]; }

    template<typename Index>
    [[nodiscard]] constexpr float3 operator[](Index i) const noexcept { return cols[i]; }
};

struct float4x4 {

    float4 cols[4];

    explicit constexpr float4x4(float s = 1.0f) noexcept
        : cols{make_float4(s, 0.0f, 0.0f, 0.0f),
               make_float4(0.0f, s, 0.0f, 0.0f),
               make_float4(0.0f, 0.0f, s, 0.0f),
               make_float4(0.0f, 0.0f, 0.0f, s)} {}

    constexpr float4x4(const float4 c0, const float4 c1, const float4 c2, const float4 c3) noexcept
        : cols{c0, c1, c2, c3} {}

    constexpr float4x4(float m00, float m01, float m02, float m03,
                       float m10, float m11, float m12, float m13,
                       float m20, float m21, float m22, float m23,
                       float m30, float m31, float m32, float m33) noexcept
        : cols{make_float4(m00, m01, m02, m03),
               make_float4(m10, m11, m12, m13),
               make_float4(m20, m21, m22, m23),
               make_float4(m30, m31, m32, m33)} {}

    template<typename Index>
    [[nodiscard]] float4 &operator[](Index i) noexcept { return cols[i]; }

    template<typename Index>
    [[nodiscard]] constexpr float4 operator[](Index i) const noexcept { return cols[i]; }
};

[[nodiscard]] constexpr auto make_float3x3(float val = 1.0f) noexcept {
    return float3x3{val};
}

[[nodiscard]] constexpr auto make_float3x3(const float3 c0, const float3 c1, const float3 c2) noexcept {
    return float3x3{c0, c1, c2};
}

[[nodiscard]] constexpr auto make_float3x3(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22) noexcept {
    return float3x3{m00, m01, m02, m10, m11, m12, m20, m21, m22};
}

[[nodiscard]] constexpr auto make_float3x3(const float4x4 m) noexcept {
    return make_float3x3(make_float3(m[0]), make_float3(m[1]), make_float3(m[2]));
}

[[nodiscard]] constexpr auto make_float4x4(float val = 1.0f) noexcept {
    return float4x4{val};
}

[[nodiscard]] constexpr auto make_float4x4(float4 c0, float4 c1, float4 c2, float4 c3) noexcept {
    return float4x4{c0, c1, c2, c3};
}

[[nodiscard]] constexpr auto make_float4x4(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33) noexcept {

    return float4x4{m00, m01, m02, m03,
                    m10, m11, m12, m13,
                    m20, m21, m22, m23,
                    m30, m31, m32, m33};
}

[[nodiscard]] constexpr auto make_float4x4(const float3x3 m) noexcept {
    return make_float4x4(
        make_float4(m[0], 0.0f),
        make_float4(m[1], 0.0f),
        make_float4(m[2], 0.0f),
        make_float4(0.0f, 0.0f, 0.0f, 1.0f));
}

[[nodiscard]] constexpr float3 operator*(const float3x3 m, float3 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2];
}

[[nodiscard]] constexpr float3x3 operator*(const float3x3 lhs, const float3x3 rhs) noexcept {
    return make_float3x3(lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]);
}

[[nodiscard]] constexpr float4 operator*(const float4x4 m, float4 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3];
}

[[nodiscard]] constexpr float4x4 operator*(const float4x4 lhs, const float4x4 rhs) noexcept {
    return make_float4x4(lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]);
}

}

}// namespace luisa

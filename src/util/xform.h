//
// Created by Mike Smith on 2022/1/15.
//

#pragma once

#include <core/basic_types.h>
#include <core/mathematics.h>

namespace luisa::render {

struct Quaternion {
    float3 v;
    float w;
    constexpr Quaternion() noexcept = default;
    constexpr Quaternion(float3 v, float w) noexcept
        : v{v}, w{w} {}
    constexpr auto operator+(Quaternion rhs) const noexcept {
        return Quaternion{v + rhs.v, w + rhs.w};
    }
    constexpr auto operator-(Quaternion rhs) const noexcept {
        return Quaternion{v - rhs.v, w - rhs.w};
    }
    constexpr auto operator*(float s) const noexcept {
        return Quaternion{v * s, w * s};
    }
    constexpr auto operator/(float s) const noexcept {
        return Quaternion{v / s, w / s};
    }
};

struct DecomposedTransform {
    float3 scaling;
    Quaternion quaternion;
    float3 translation;
};

[[nodiscard]] DecomposedTransform decompose(float4x4 m) noexcept;
[[nodiscard]] Quaternion quaternion(float3x3 m) noexcept;
[[nodiscard]] float4x4 rotation(Quaternion q) noexcept;
[[nodiscard]] float dot(Quaternion q1, Quaternion q2) noexcept;
[[nodiscard]] float length(Quaternion q) noexcept;
[[nodiscard]] float angle_between(Quaternion q1, Quaternion q2) noexcept;
[[nodiscard]] Quaternion normalize(Quaternion q) noexcept;
[[nodiscard]] Quaternion slerp(Quaternion q1, Quaternion q2, float t) noexcept;

}// namespace luisa::render

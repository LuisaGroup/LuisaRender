//
// Created by Mike Smith on 2022/1/15.
//

#include <array>

#include <core/logging.h>
#include <util/xform.h>

namespace luisa::render {

DecomposedTransform decompose(float4x4 m) noexcept {
    auto t = m[3].xyz();
    auto N = make_float3x3(m);
    auto near_zero = [](auto f) noexcept {
        return std::abs(f) <= 1e-4f;
    };
    auto R = N;
    constexpr auto max_iteration_count = 100u;
    for (auto i = 0u; i < max_iteration_count; i++) {
        auto R_it = inverse(transpose(R));
        auto R_next = 0.5f * (R + R_it);
        auto diff = R - R_next;
        R = R_next;
        auto n = abs(diff[0]) + abs(diff[1]) + abs(diff[2]);
        if (near_zero(std::max({n.x, n.y, n.z}))) { break; }
    }
    auto S = inverse(R) * N;
    if (!near_zero(S[0].y) || !near_zero(S[0].z) ||
        !near_zero(S[1].x) || !near_zero(S[1].z) ||
        !near_zero(S[2].x) || !near_zero(S[2].y)) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Non-zero entries found in decomposed scaling matrix: "
            "(({}, {}, {}), ({}, {}, {}), ({}, {}, {})).",
            S[0].x, S[1].x, S[2].x,
            S[0].y, S[1].y, S[2].y,
            S[0].z, S[1].z, S[2].z);
    }
    auto s = make_float3(S[0].x, S[1].y, S[2].z);
    auto q = quaternion(R);
    return {s, q, t};
}

Quaternion quaternion(float3x3 m) noexcept {
    if (auto trace = m[0][0] + m[1][1] + m[2][2];
        trace > 0.0f) {
        auto s = std::sqrt(trace + 1.0f);
        auto w = 0.5f * s;
        s = 0.5f / s;
        auto v = make_float3(
            (m[1][2] - m[2][1]),
            (m[2][0] - m[0][2]),
            (m[0][1] - m[1][0]));
        return {v * s, w};
    }
    std::array next{1u, 2u, 0u};
    float3 v;
    auto i = 0u;
    if (m[1][1] > m[0][0]) { i = 1u; }
    if (m[2][2] > m[i][i]) { i = 2u; }
    auto j = next[i];
    auto k = next[j];
    auto s = std::sqrt(std::max(
        m[i][i] - (m[j][j] + m[k][k]) + 1.0f,
        0.0f));
    v[i] = s * 0.5f;
    if (s != 0.0f) { s = 0.5f / s; }
    auto w = (m[j][k] - m[k][j]) * s;
    v[j] = (m[i][j] + m[j][i]) * s;
    v[k] = (m[i][k] + m[k][i]) * s;
    return {v, w};
}

float4x4 rotation(Quaternion q) noexcept {
    auto theta = 2.0f * std::atan2(length(q.v), q.w);
    return rotation(normalize(q.v), theta);
}

float angle_between(Quaternion q1, Quaternion q2) noexcept {
    auto safe_asin = [](auto x) noexcept {
        return std::asin(std::clamp(x, -1.0f, 1.0f));
    };
    return dot(q1, q2) < 0.0f ?
               pi - 2.0f * safe_asin(length(q1 + q2) * 0.5f) :
               2.0f * safe_asin(length(q1 - q2) * 0.5f);
}

Quaternion slerp(Quaternion q1, Quaternion q2, float t) noexcept {
    auto sin_x_over_x = [](auto x) noexcept {
        return 1.0f + x * x == 1.0f ? 1.0f : std::sin(x) / x;
    };
    auto theta = angle_between(q1, q2);
    auto sin_theta_over_theta = sin_x_over_x(theta);
    return normalize(
        q1 * (1.0f - t) * sin_x_over_x((1.0f - t) * theta) / sin_theta_over_theta +
        q2 * t * sin_x_over_x(t * theta) / sin_theta_over_theta);
}

float dot(Quaternion q1, Quaternion q2) noexcept {
    return dot(q1.v, q2.v) + q1.w * q2.w;
}

float length(Quaternion q) noexcept {
    return std::sqrt(dot(q, q));
}

Quaternion normalize(Quaternion q) noexcept {
    return q / length(q);
}

}// namespace luisa::render

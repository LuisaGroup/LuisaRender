//
// Created by Mike Smith on 2022/11/8.
//

#pragma once

#include <concepts>
#include <dsl/syntax.h>
#include <rtx/mesh.h>

namespace luisa::render {

template<typename T>
    requires std::same_as<luisa::compute::expr_value_t<T>, float3>
[[nodiscard]] inline auto oct_encode(T n) noexcept {
    constexpr auto oct_wrap = [](auto v) noexcept {
        return (1.f - abs(v.yx())) * select(make_float2(-1.f), make_float2(1.f), v >= 0.0f);
    };
    auto abs_n = abs(n);
    auto p = n.xy() * (1.f / (abs_n.x + abs_n.y + abs_n.z));
    p = select(oct_wrap(p), p, n.z >= 0.f);// in [-1, 1]
    auto u = make_uint2(clamp(round((p * .5f + .5f) * 65535.f), 0.f, 65535.f));
    return u.x | (u.y << 16u);
};

template<typename T>
    requires std::same_as<luisa::compute::expr_value_t<T>, uint>
[[nodiscard]] inline auto oct_decode(T u) noexcept {
    auto p = make_float2(make_uint2(u & 0xffffu, u >> 16u)) * ((1.f / 65535.f) * 2.f) - 1.f;
    auto abs_p = abs(p);
    auto n = make_float3(p, 1.f - abs_p.x - abs_p.y);
    auto t = make_float2(clamp(-n.z, 0.f, 1.f));
    return normalize(make_float3(n.xy() + select(t, -t, n.xy() >= 0.f), n.z));
}

template<typename T>
    requires std::same_as<luisa::compute::expr_value_t<T>, float3>
[[nodiscard]] inline auto rgb_encode(T c) noexcept {
    auto u = make_uint3(clamp(round(c * 255.f), 0.f, 255.f));
    return u.x | (u.y << 8u) | (u.z << 16u);
};

template<typename T>
    requires std::same_as<luisa::compute::expr_value_t<T>, uint>
[[nodiscard]] inline auto rgb_decode(T c) noexcept {
    auto rgb_u8 = make_uint3(c & 0xffu, (c >> 8u) & 0xffu, (c >> 16u) & 0xffu);
    return make_float3(rgb_u8) * (1.f / 255.f);
}

struct alignas(16) Vertex {

    float px;
    float py;
    float pz;
    uint rgb;
    uint n;
    uint s;
    float u;
    float v;

    [[nodiscard]] static auto encode(float3 position, float3 color,
                                     float3 normal, float3 tangent, float2 uv) noexcept {
        return Vertex{position.x, position.y, position.z, rgb_encode(color),
                      oct_encode(normal), oct_encode(tangent), uv.x, uv.y};
    };

    [[nodiscard]] auto position() const noexcept { return make_float3(px, py, pz); }
    [[nodiscard]] auto color() const noexcept { return rgb_decode(rgb); }
    [[nodiscard]] auto normal() const noexcept { return oct_decode(n); }
    [[nodiscard]] auto tangent() const noexcept { return oct_decode(s); }
    [[nodiscard]] auto uv() const noexcept { return make_float2(u, v); }
};

[[nodiscard]] float3 compute_tangent(float3 p0, float3 p1, float3 p2,
                                     float2 uv0, float2 uv1, float2 uv2) noexcept;
void compute_tangents(luisa::span<Vertex> vertices,
                      luisa::span<const compute::Triangle> triangles,
                      bool area_weighted = true) noexcept;

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::Vertex, px, py, pz, rgb, n, s, u, v) {

    [[nodiscard]] static auto encode(luisa::compute::Expr<luisa::float3> position,
                                     luisa::compute::Expr<luisa::float3> color,
                                     luisa::compute::Expr<luisa::float3> normal,
                                     luisa::compute::Expr<luisa::float3> tangent,
                                     luisa::compute::Expr<luisa::float2> uv) noexcept {
        return def<luisa::render::Vertex>(position.x, position.y, position.z,
                                          luisa::render::rgb_encode(color),
                                          luisa::render::oct_encode(normal),
                                          luisa::render::oct_encode(tangent),
                                          uv.x, uv.y);
    };
    [[nodiscard]] auto position() const noexcept { return make_float3(px, py, pz); }
    [[nodiscard]] auto color() const noexcept { return luisa::render::rgb_decode(rgb); }
    [[nodiscard]] auto normal() const noexcept { return luisa::render::oct_decode(n); }
    [[nodiscard]] auto tangent() const noexcept { return luisa::render::oct_decode(s); }
    [[nodiscard]] auto uv() const noexcept { return make_float2(u, v); }
};
// clang-format on

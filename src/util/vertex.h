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
[[nodiscard]] inline auto oct_encode(T n_in) noexcept {
    auto n = select(normalize(n_in), make_float3(0.f, 0.f, 1.f), all(n_in == 0.f));
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

struct alignas(16) Vertex {

    float px;
    float py;
    float pz;
    uint n;

    [[nodiscard]] static auto encode(float3 position, float3 normal) noexcept {
        return Vertex{position.x, position.y, position.z, oct_encode(normal)};
    };
    [[nodiscard]] auto position() const noexcept { return make_float3(px, py, pz); }
    [[nodiscard]] auto normal() const noexcept { return oct_decode(n); }
};

static_assert(sizeof(Vertex) == 16u);

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::Vertex, px, py, pz, n) {
    [[nodiscard]] static auto encode(luisa::compute::Expr<luisa::float3> position,
                                     luisa::compute::Expr<luisa::float3> normal) noexcept {
        return def<luisa::render::Vertex>(position.x, position.y, position.z,
                                          luisa::render::oct_encode(normal));
    };
    [[nodiscard]] auto position() const noexcept { return make_float3(px, py, pz); }
    [[nodiscard]] auto normal() const noexcept { return luisa::render::oct_decode(n); }
};
// clang-format on

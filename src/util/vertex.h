//
// Created by Mike Smith on 2022/11/8.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

struct alignas(16) Vertex {
    std::array<float, 3> compressed_p;
    std::array<float, 3> compressed_n;
    std::array<float, 2> compressed_uv;
    [[nodiscard]] static auto encode(float3 position, float3 normal, float2 uv) noexcept {
        return Vertex{
            .compressed_p = {position.x, position.y, position.z},
            .compressed_n = {normal.x, normal.y, normal.z},
            .compressed_uv = {uv.x, uv.y}};
    };
};

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::Vertex, compressed_p, compressed_n, compressed_uv) {
    [[nodiscard]] auto position() const noexcept { return make_float3(compressed_p[0], compressed_p[1], compressed_p[2]); }
    [[nodiscard]] auto normal() const noexcept { return make_float3(compressed_n[0], compressed_n[1], compressed_n[2]); }
    [[nodiscard]] auto uv() const noexcept { return make_float2(compressed_uv[0], compressed_uv[1]); }
};
// clang-format on

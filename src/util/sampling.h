//
// Created by Mike Smith on 2022/1/9.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using luisa::compute::Expr;
using luisa::compute::Float;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::UInt;
using luisa::compute::Var;

[[nodiscard]] Float3 sample_uniform_triangle(Expr<float2> u) noexcept;
[[nodiscard]] Float2 sample_uniform_disk_concentric(Expr<float2> u) noexcept;
[[nodiscard]] Float3 sample_cosine_hemisphere(Expr<float2> u) noexcept;
[[nodiscard]] Float cosine_hemisphere_pdf(Expr<float> cos_theta) noexcept;
[[nodiscard]] Float3 sample_uniform_sphere(Expr<float2> u) noexcept;
[[nodiscard]] constexpr float uniform_sphere_pdf() noexcept { return inv_pi * 0.25f; }
[[nodiscard]] Float2 invert_uniform_sphere_sample(Expr<float3> w) noexcept;

struct AliasEntry {
    float prob;
    uint alias;
};

// reference: https://github.com/AirGuanZ/agz-utils
[[nodiscard]] std::pair<luisa::vector<AliasEntry>, luisa::vector<float>/* pdf */>
    create_alias_table(luisa::span<float> values) noexcept;

template<typename Table>
[[nodiscard]] auto sample_alias_table(const Table &table, Expr<uint> n, Expr<float> u_in) noexcept {
    using namespace luisa::compute;
    auto u = u_in * cast<float>(n);
    auto i = clamp(cast<uint>(u), 0u, n - 1u);
    auto u_remapped = u - cast<float>(i);
    auto entry = table.read(i);
    return ite(u_remapped < entry.prob, i, entry.alias);
}

}// namespace luisa::render

LUISA_STRUCT(luisa::render::AliasEntry, prob, alias){};

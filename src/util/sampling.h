//
// Created by Mike Smith on 2022/1/9.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using luisa::compute::Expr;
using luisa::compute::Var;
using luisa::compute::Float;
using luisa::compute::Float2;
using luisa::compute::Float3;

[[nodiscard]] Float2 sample_uniform_disk_concentric(Expr<float2> u) noexcept;
[[nodiscard]] Float3 sample_cosine_hemisphere(Expr<float2> u) noexcept;
[[nodiscard]] Float cosine_hemisphere_pdf(Expr<float> cos_theta) noexcept;

}

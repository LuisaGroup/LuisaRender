//
// Created by Mike Smith on 2022/1/19.
//

#pragma once

#include <core/basic_types.h>
#include <dsl/syntax.h>

namespace luisa::render {

using luisa::compute::Expr;
using luisa::compute::Float3;
using luisa::compute::Float4;

// TODO: other colorspaces than sRGB
[[nodiscard]] Float3 cie_xyz_to_linear_srgb(Expr<float3> xyz) noexcept;

}

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

class SampledWavelengths;

[[nodiscard]] Float4 sample_cie_x(const SampledWavelengths &lambda) noexcept;
[[nodiscard]] Float4 sample_cie_y(const SampledWavelengths &lambda) noexcept;
[[nodiscard]] Float4 sample_cie_z(const SampledWavelengths &lambda) noexcept;

// TODO: other colorspaces than sRGB
[[nodiscard]] Float3 cie_xyz_to_rgb(Expr<float3> xyz) noexcept;

}

//
// Created by Mike Smith on 2022/1/27.
//

#pragma once

#include <core/basic_types.h>

namespace luisa::render {

constexpr auto half_max = 65504.0f;
constexpr auto half_min = -65504.0f;

[[nodiscard]] uint float_to_half(float f) noexcept;
[[nodiscard]] float half_to_float(uint h) noexcept;

}// namespace luisa::render

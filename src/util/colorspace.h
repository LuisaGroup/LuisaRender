//
// Created by Mike Smith on 2022/1/19.
//

#pragma once

#include <core/basic_types.h>
#include <dsl/syntax.h>

namespace luisa::render {

template<typename T>
[[nodiscard]] auto cie_xyz_to_linear_srgb(T &&xyz) noexcept {
    constexpr auto m = make_float3x3(
        +3.240479f, -0.969256f, +0.055648f,
        -1.537150f, +1.875991f, -0.204043f,
        -0.498535f, +0.041556f, +1.057311f);
    return m * std::forward<T>(xyz);
}

template<typename T>
[[nodiscard]] auto srgb_to_cie_y(T &&rgb) noexcept {
    auto m = make_float3(0.212671f, 0.715160f, 0.072169f);
    return dot(m, std::forward<T>(rgb));
}

}// namespace luisa::render

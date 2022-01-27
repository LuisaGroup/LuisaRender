//
// Created by Mike Smith on 2022/1/27.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

namespace detail {

[[nodiscard]] inline auto float_bits_to_half(auto x) noexcept {
    return ((x >> 16u) & 0x8000u) |
           ((((x & 0x7f800000u) - 0x38000000u) >> 13u) & 0x7c00u) |
           ((x >> 13u) & 0x03ffu);
}

[[nodiscard]] inline auto half_to_float_bits(auto h) noexcept {
    return ((h & 0x8000u) << 16u) |
           (((h & 0x7c00u) + 0x1c000u) << 13u) |
           ((h & 0x03ffu) << 13u);
}

}

using compute::Expr;
using compute::UInt;
using compute::Float;
using compute::as;

// assumes IEEE-754
[[nodiscard]] inline uint float_to_half(float f) noexcept {
    auto x = luisa::bit_cast<uint>(f);
    return detail::float_bits_to_half(x);
}

[[nodiscard]] inline float half_to_float(uint h) noexcept {
    auto x = detail::half_to_float_bits(h);
    return luisa::bit_cast<float>(x);
}

[[nodiscard]] inline UInt float_to_half(Expr<float> f) noexcept {
    auto x = as<uint>(f);
    return detail::float_bits_to_half(x);
}

[[nodiscard]] inline Float half_to_float(Expr<uint> h) noexcept {
    auto x = detail::half_to_float_bits(h);
    return as<float>(x);
}

}// namespace luisa::render

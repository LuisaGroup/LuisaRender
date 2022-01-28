//
// Created by Mike Smith on 2022/1/27.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

constexpr auto half_max = 65504.0f;
constexpr auto half_min = -65504.0f;

namespace detail {

[[nodiscard]] inline auto half_to_float_bits(auto h) noexcept {
    return ((h & 0x8000u) << 16u) |
           (((h & 0x7c00u) + 0x1c000u) << 13u) |
           ((h & 0x03ffu) << 13u);
}

}// namespace detail

using compute::as;
using compute::Expr;
using compute::Float;
using compute::UInt;

// from tinyexr: https://github.com/syoyo/tinyexr/blob/master/tinyexr.h
[[nodiscard]] inline uint float_to_half(float f) noexcept {
    auto bits = luisa::bit_cast<uint>(f);
    auto fp32_sign = bits >> 31u;
    auto fp32_exponent = (bits >> 23u) & 0xffu;
    auto fp32_mantissa = bits & ((1u << 23u) - 1u);

    auto make_fp16 = [](uint sign, uint exponent, uint mantissa) noexcept {
        return (sign << 15u) | (exponent << 10u) | mantissa;
    };
    // Signed zero/denormal (which will underflow)
    if (fp32_exponent == 0u) { return make_fp16(fp32_sign, 0u, 0u); }

    // Inf or NaN (all exponent bits set)
    if (fp32_exponent == 255u) {
        return make_fp16(
            fp32_sign, 31u,
            // NaN->qNaN and Inf->Inf
            fp32_mantissa ? 0x200u : 0u);
    }

    // Exponent unbias the single, then bias the halfp
    auto newexp = static_cast<int>(fp32_exponent - 127u + 15u);

    // Overflow, return signed infinity
    if (newexp >= 31) { return make_fp16(fp32_sign, 31u, 0u); }

    // Underflow
    if (newexp <= 0) {
        if ((14 - newexp) > 24) { return 0u; }
        // Mantissa might be non-zero
        unsigned int mant = fp32_mantissa | 0x800000u;// Hidden 1 bit
        auto fp16 = make_fp16(fp32_sign, 0u, mant >> (14u - newexp));
        if ((mant >> (13u - newexp)) & 1u) { fp16++; }// Check for rounding
        return fp16;
    }

    auto fp16 = make_fp16(fp32_sign, newexp, fp32_mantissa >> 13u);
    if (fp32_mantissa & 0x1000u) { fp16++; }// Check for rounding
    return fp16;
}

[[nodiscard]] inline float half_to_float(uint h) noexcept {
    auto x = detail::half_to_float_bits(h);
    return luisa::bit_cast<float>(x);
}

[[nodiscard]] inline Float half_to_float(Expr<uint> h) noexcept {
    auto x = detail::half_to_float_bits(h);
    return as<float>(x);
}

}// namespace luisa::render

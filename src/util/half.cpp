//
// Created by Mike Smith on 2022/11/18.
//

#include <core/stl.h>
#include <util/half.h>

namespace luisa::render {

// from tinyexr: https://github.com/syoyo/tinyexr/blob/master/tinyexr.h
float half_to_float(uint h) noexcept {
    union FP32 {
        uint u;
        float f;
        struct {// FIXME: assuming little endian here
            uint Mantissa : 23;
            uint Exponent : 8;
            uint Sign : 1;
        } s;
    };
    constexpr auto magic = FP32{113u << 23u};
    constexpr auto shifted_exp = 0x7c00u << 13u;// exponent mask after shift
    auto o = FP32{(h & 0x7fffu) << 13u};        // exponent/mantissa bits
    auto exp_ = shifted_exp & o.u;              // just the exponent
    o.u += (127u - 15u) << 23u;                 // exponent adjust

    // handle exponent special cases
    if (exp_ == shifted_exp) {     // Inf/NaN?
        o.u += (128u - 16u) << 23u;// extra exp adjust
    } else if (exp_ == 0u) {       // Zero/Denormal?
        o.u += 1u << 23u;          // extra exp adjust
        o.f -= magic.f;            // renormalize
    }
    o.u |= (h & 0x8000u) << 16u;// sign bit
    return o.f;
}

uint float_to_half(float f) noexcept {
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
        auto mant = fp32_mantissa | 0x800000u;// Hidden 1 bit
        auto fp16 = make_fp16(fp32_sign, 0u, mant >> (14u - newexp));
        if ((mant >> (13u - newexp)) & 1u) { fp16++; }// Check for rounding
        return fp16;
    }

    auto fp16 = make_fp16(fp32_sign, newexp, fp32_mantissa >> 13u);
    if (fp32_mantissa & 0x1000u) { fp16++; }// Check for rounding
    return fp16;
}

}// namespace luisa::render

//
// Created by Mike Smith on 2022/2/9.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using compute::cast;
using compute::Expr;
using compute::make_uint2;
using compute::UInt2;

[[nodiscard]] constexpr auto u64_to_uint2(uint64_t x) noexcept {
    return luisa::make_uint2(
        static_cast<uint>(x >> 32u) /* hi */,
        static_cast<uint>(x) /* lo */);
}

[[nodiscard]] constexpr auto uint2_to_u64(uint2 v) noexcept {
    return (static_cast<uint64_t>(v.x) << 32u) | v.y;
}

class U64 {

private:
    UInt2 _bits;

private:
    [[nodiscard]] static auto _mul_u32(Expr<uint> lhs, Expr<uint> rhs) noexcept {
        auto lhs_hi = lhs >> 16u;
        auto lhs_lo = lhs & 0xffffu;
        auto rhs_hi = rhs >> 16u;
        auto rhs_lo = rhs & 0xffffu;
        auto hi_lo = lhs_hi * rhs_lo;
        auto lo_lo = lhs_lo * rhs_lo;
        auto lo_hi = lhs_lo * rhs_hi;
        auto hi_hi = lhs_hi * rhs_hi;
        auto m_16_32 = (lo_lo >> 16u) + (hi_lo & 0xffffu) + (lo_hi & 0xffffu);
        auto m_32_64 = (m_16_32 >> 16u) + (hi_lo >> 16u) + (lo_hi >> 16u) + hi_hi;
        return U64{m_32_64, (m_16_32 << 16u) | (lo_lo & 0xffffu)};
    }

public:
    explicit U64(uint64_t u = 0ull) noexcept : _bits{u64_to_uint2(u)} {}
    explicit U64(Expr<uint2> u) noexcept : _bits{u} {}
    explicit U64(Expr<uint> u) noexcept : _bits{make_uint2(0u, u)} {}
    U64(Expr<uint> hi, Expr<uint> lo)
    noexcept : _bits{make_uint2(hi, lo)} {}
    U64(U64 &&)
    noexcept = default;
    U64(const U64 &)
    noexcept = default;
    U64 &operator=(U64 &&) noexcept = default;
    U64 &operator=(const U64 &) noexcept = default;
    [[nodiscard]] auto hi() const noexcept { return _bits.x; }
    [[nodiscard]] auto lo() const noexcept { return _bits.y; }
    [[nodiscard]] auto bits() const noexcept { return _bits; }
    [[nodiscard]] auto operator~() const noexcept { return U64{~_bits}; }
    [[nodiscard]] auto operator&(Expr<uint> rhs) const noexcept { return lo() & rhs; }
    [[nodiscard]] auto operator&(const U64 &rhs) const noexcept { return U64{_bits & rhs._bits}; }
    [[nodiscard]] auto operator|(Expr<uint> rhs) const noexcept { return U64{hi(), lo() | rhs}; }
    [[nodiscard]] auto operator|(const U64 &rhs) const noexcept { return U64{_bits | rhs._bits}; }
    [[nodiscard]] auto operator^(Expr<uint> rhs) const noexcept { return U64{hi(), lo() ^ rhs}; }
    [[nodiscard]] auto operator^(const U64 &rhs) const noexcept { return U64{_bits ^ rhs._bits}; }
    // TODO: optimize this
    [[nodiscard]] auto operator>>(Expr<uint> rhs) const noexcept {
        using compute::if_;
        auto ret = *this;
        if_(rhs != 0u, [&] {
            if_(rhs >= 32u, [&] {
                ret = U64{0u, hi() >> (rhs - 32u)};
            }).else_([&] {
                ret = U64{hi() >> rhs, (hi() << (32u - rhs)) | (lo() >> rhs)};
            });
        });
        return ret;
    }
    [[nodiscard]] auto operator<<(Expr<uint> rhs) const noexcept {
        using compute::if_;
        auto ret = *this;
        if_(rhs != 0u, [&] {
            if_(rhs >= 32u, [&] {
                ret = U64{lo() << (rhs - 32u), 0u};
            }).else_([&] {
                ret = U64{(hi() << rhs) | (lo() >> (32u - rhs)), lo() << rhs};
            });
        });
        return ret;
    }
    [[nodiscard]] auto operator==(const U64 &rhs) const noexcept { return all(_bits == rhs._bits); }
    [[nodiscard]] auto operator==(Expr<uint> rhs) const noexcept { return hi() == 0u & lo() == rhs; }
    [[nodiscard]] auto operator!=(const U64 &rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] auto operator!=(Expr<uint> rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] auto operator+(const U64 &rhs) const noexcept -> U64 {
        auto carry = cast<uint>(~0u - lo() < rhs.lo());
        return U64{hi() + rhs.hi() + carry, lo() + rhs.lo()};
    }
    [[nodiscard]] auto operator+(Expr<uint> rhs) const noexcept -> U64 {
        auto carry = cast<uint>(~0u - lo() < rhs);
        return U64{hi() + carry, lo() + rhs};
    }
    [[nodiscard]] auto operator-(const U64 &rhs) const noexcept {
        return *this + ~rhs + 1u;
    }
    [[nodiscard]] auto operator-(Expr<uint> rhs) const noexcept {
        return *this - U64{rhs};
    }
    [[nodiscard]] auto operator*(const U64 &rhs) const noexcept {
        auto lo_lo = _mul_u32(lo(), rhs.lo());
        auto lo_hi = _mul_u32(lo(), rhs.hi());
        auto hi_lo = _mul_u32(hi(), rhs.lo());
        return U64{lo_lo.hi() + lo_hi.lo() + hi_lo.lo(), lo_lo.lo()};
    }
    [[nodiscard]] auto operator*(Expr<uint> rhs) const noexcept {
        auto lo_lo = _mul_u32(lo(), rhs);
        auto hi_lo = _mul_u32(hi(), rhs);
        return U64{lo_lo.hi() + hi_lo.lo(), lo_lo.lo()};
    }
    [[nodiscard]] auto operator%(uint rhs) const noexcept {
        LUISA_ASSERT(rhs <= 0xffffu, "U64::operator% rhs must be <= 0xffff");
        return ((hi() % rhs) * static_cast<uint>(0x1'0000'0000ull % rhs) + lo() % rhs) % rhs;
    }
};

}// namespace luisa::render

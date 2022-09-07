//
// Created by Mike Smith on 2022/8/31.
//

#include <random>
#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

[[nodiscard]] constexpr auto u64_to_uint2(uint64_t x) noexcept {
    return luisa::make_uint2(
        static_cast<uint>(x >> 32u) /* hi */,
        static_cast<uint>(x) /* lo */);
}

[[nodiscard]] constexpr auto uint2_to_u64(uint2 v) noexcept {
    //    LUISA_INFO("{} {}", v.x, v.y);
    auto r = (static_cast<uint64_t>(v.x) << 32u) | v.y;
    //    LUISA_INFO("R = {}", r);
    return r;
}

class U64 {

private:
    uint2 _bits;

public:
    [[nodiscard]] static auto _mul_u32(uint lhs, uint rhs) noexcept {
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
    explicit U64(uint2 u) noexcept : _bits{u} {}
    explicit U64(uint u) noexcept : _bits{make_uint2(0u, u)} {}
    U64(uint hi, uint lo)
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
    [[nodiscard]] auto operator&(uint rhs) const noexcept { return lo() & rhs; }
    [[nodiscard]] auto operator&(const U64 &rhs) const noexcept { return U64{_bits & rhs._bits}; }
    [[nodiscard]] auto operator|(uint rhs) const noexcept { return U64{hi(), lo() | rhs}; }
    [[nodiscard]] auto operator|(const U64 &rhs) const noexcept { return U64{_bits | rhs._bits}; }
    [[nodiscard]] auto operator^(uint rhs) const noexcept { return U64{hi(), lo() ^ rhs}; }
    [[nodiscard]] auto operator^(const U64 &rhs) const noexcept { return U64{_bits ^ rhs._bits}; }
    [[nodiscard]] auto operator>>(uint rhs) const noexcept {
        if (rhs == 0u) { return *this; }
        if (rhs >= 32u) { return U64{0u, hi() >> (rhs - 32u)}; }
        return U64{hi() >> rhs, (hi() << (32u - rhs)) | (lo() >> rhs)};
    }
    [[nodiscard]] auto operator<<(uint rhs) const noexcept {
        if (rhs == 0u) { return *this; }
        if (rhs >= 32u) { return U64{lo() << (rhs - 32u), 0u}; }
        return U64{(hi() << rhs) | (lo() >> (32u - rhs)), lo() << rhs};
    }
    [[nodiscard]] auto operator==(const U64 &rhs) const noexcept { return all(_bits == rhs._bits); }
    [[nodiscard]] auto operator==(uint rhs) const noexcept { return hi() == 0u & lo() == rhs; }
    [[nodiscard]] auto operator!=(const U64 &rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] auto operator!=(uint rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] auto operator+(const U64 &rhs) const noexcept -> U64 {
        auto carry = cast<uint>(~0u - lo() < rhs.lo());
        return U64{hi() + rhs.hi() + carry, lo() + rhs.lo()};
    }
    [[nodiscard]] auto operator+(uint rhs) const noexcept -> U64 {
        auto carry = cast<uint>(~0u - lo() < rhs);
        return U64{hi() + carry, lo() + rhs};
    }
    [[nodiscard]] auto operator-(const U64 &rhs) const noexcept {
        return *this + ~rhs + 1u;
    }
    [[nodiscard]] auto operator-(uint rhs) const noexcept {
        return *this - U64{rhs};
    }
    [[nodiscard]] auto operator*(const U64 &rhs) const noexcept {
        auto lo_lo = _mul_u32(lo(), rhs.lo());
        auto lo_hi = _mul_u32(lo(), rhs.hi());
        auto hi_lo = _mul_u32(hi(), rhs.lo());
        return U64{lo_lo.hi() + lo_hi.lo() + hi_lo.lo(), lo_lo.lo()};
    }
    [[nodiscard]] auto operator*(uint rhs) const noexcept {
        auto lo_lo = _mul_u32(lo(), rhs);
        auto hi_lo = _mul_u32(hi(), rhs);
        return U64{lo_lo.hi() + hi_lo.lo(), lo_lo.lo()};
    }
    [[nodiscard]] auto operator%(uint rhs) const noexcept {
        LUISA_ASSERT(rhs <= 0xffffu, "U64::operator% rhs must be <= 0xffff");
        return ((hi() % rhs) * static_cast<uint>(0x1'0000'0000ull % rhs) + lo() % rhs) % rhs;
    }
};

int main() {

    static constexpr auto N = 100'000'000ull;
    std::mt19937_64 rng{std::random_device{}()};

    for (auto i = 0u; i < N; i++) {
        auto x = rng();
        auto y = rng();
        auto xx = U64{x};
        auto yy = U64{y};
#define TEST_L(op)                                                 \
    {                                                              \
        LUISA_ASSERT(uint2_to_u64((xx op yy).bits()) == (x op y),  \
                     "Error #{}: 0x{:x} " #op " 0x{:x}", i, x, y); \
    }
        TEST_L(|);
        TEST_L(&);
        TEST_L(^);
        TEST_L(+);
        TEST_L(-);
        TEST_L(*);
        auto z = static_cast<uint>(y);
#define TEST_U(op)                                        \
    {                                                     \
        auto got = uint2_to_u64((xx op z).bits());        \
        auto expt = x op z;                               \
        LUISA_ASSERT(expt == got,                         \
                     "Error #{}: 0x{:x} " #op " 0x{:x}, " \
                     "expected 0x{:x}, got 0x{:x}",       \
                     i, x, z, expt, got);                 \
    }
        TEST_U(|);
        TEST_U(^);
        TEST_U(+);
        TEST_U(-);
        if (z == 0u) { z = 1u; }
        {
            auto t = x & ~0u;
            auto got = uint2_to_u64(U64::_mul_u32(t, z).bits());
            auto expt = t * z;
            LUISA_ASSERT(got == expt,
                         "Error #{}: 0x{:x} * 0x{:x}, "
                         "expected 0x{:x}, got 0x{:x}",
                         i, x, z, expt, got);
        }
        TEST_U(*);
        z = z & 0xffffu;
        if (z == 0u) { z = 1u; }
        {
            LUISA_ASSERT((xx % z) == (x % z),
                         "Error #{}: 0x{:x} % 0x{:x}", i, x, z);
        }
        z = z & 63u;
        TEST_U(>>);
        TEST_U(<<);
    }
}

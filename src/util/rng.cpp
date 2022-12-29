//
// Created by Mike Smith on 2022/2/12.
//

#include <util/rng.h>

namespace luisa::render {

using compute::Callable;
using compute::def;

UInt xxhash32(Expr<uint> p) noexcept {
    static Callable impl = [](UInt p) noexcept {
        constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
        constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
        auto h32 = p + PRIME32_5;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
        h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
        return h32 ^ (h32 >> 16u);
    };
    return impl(p);
}

UInt xxhash32(Expr<uint2> p) noexcept {
    static Callable impl = [](UInt2 p) noexcept {
        constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
        constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
        auto h32 = p.y + PRIME32_5 + p.x * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
        h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
        return h32 ^ (h32 >> 16u);
    };
    return impl(p);
}

UInt xxhash32(Expr<uint3> p) noexcept {
    static Callable impl = [](UInt3 p) noexcept {
        constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
        constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
        UInt h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 += p.y * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
        h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
        return h32 ^ (h32 >> 16u);
    };
    return impl(p);
}

UInt xxhash32(Expr<uint4> p) noexcept {
    static Callable impl = [](UInt4 p) noexcept {
        constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
        constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
        auto h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 += p.y * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 += p.z * PRIME32_3;
        h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
        h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
        h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
        return h32 ^ (h32 >> 16u);
    };
    return impl(p);
}

// https://www.pcg-random.org/
UInt pcg(Expr<uint> v) noexcept {
    static Callable impl = [](UInt v) noexcept {
        auto state = v * 747796405u + 2891336453u;
        auto word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    };
    return impl(v);
}

UInt2 pcg2d(Expr<uint2> v) noexcept {
    static Callable impl = [](UInt2 v) noexcept {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * 1664525u;
        v.y += v.x * 1664525u;
        v = v ^ (v >> 16u);
        v.x += v.y * 1664525u;
        v.y += v.x * 1664525u;
        v = v ^ (v >> 16u);
        return v;
    };
    return impl(v);
}

// http://www.jcgt.org/published/0009/03/02/
UInt3 pcg3d(Expr<uint3> v) noexcept {
    static Callable impl = [](UInt3 v) noexcept {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.z;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v ^= v >> 16u;
        v.x += v.y * v.z;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        return v;
    };
    return impl(v);
}

// http://www.jcgt.org/published/0009/03/02/
UInt4 pcg4d(Expr<uint4> v) noexcept {
    static Callable impl = [](UInt4 v) noexcept {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.w;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v.w += v.y * v.z;
        v ^= v >> 16u;
        v.x += v.y * v.w;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v.w += v.y * v.z;
        return v;
    };
    return impl(v);
}

Float uniform_uint_to_float(Expr<uint> u) noexcept {
    return min(one_minus_epsilon, u * 0x1p-32f);
}

Float lcg(UInt &state) noexcept {
    static Callable impl = [](UInt &state) noexcept {
        constexpr auto lcg_a = 1664525u;
        constexpr auto lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return uniform_uint_to_float(state);
    };
    return impl(state);
}

UInt PCG32::uniform_uint() noexcept {
    auto oldstate = _state;
    _state = oldstate * U64{mult} + _inc;
    auto xorshifted = (((oldstate >> 18u) ^ oldstate) >> 27u).lo();
    auto rot = (oldstate >> 59u).lo();
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
}

void PCG32::set_sequence(U64 init_seq) noexcept {
    _state = U64{0u};
    _inc = (init_seq << 1u) | 1u;
    static_cast<void>(uniform_uint());
    _state = _state + U64{default_state};
    static_cast<void>(uniform_uint());
}

Float PCG32::uniform_float() noexcept {
    return uniform_uint_to_float(uniform_uint());
}

PCG32::PCG32() noexcept
    : _state{default_state}, _inc{default_stream} {}

PCG32::PCG32(U64 state, U64 inc) noexcept
    : _state{std::move(state)}, _inc{std::move(inc)} {}

PCG32::PCG32(U64 seq_index) noexcept {
    set_sequence(std::move(seq_index));
}

PCG32::PCG32(Expr<uint> seq_index) noexcept {
    set_sequence(U64{seq_index});
}

}// namespace luisa::render

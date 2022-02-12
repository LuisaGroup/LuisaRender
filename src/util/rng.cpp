//
// Created by Mike Smith on 2022/2/12.
//

#include <util/rng.h>

namespace luisa::render {

using compute::def;

UInt xxhash32(Expr<uint> p) noexcept {
    constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    auto h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
    return h32 ^ (h32 >> 16u);
}

UInt xxhash32(Expr<uint2> p) noexcept {
    constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    auto h32 = p.y + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
    return h32 ^ (h32 >> 16u);
}

UInt xxhash32(Expr<uint3> p) noexcept {
    constexpr auto PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr auto PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    UInt h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> (32u - 17u)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
    return h32 ^ (h32 >> 16u);
}

UInt xxhash32(Expr<uint4> p) noexcept {
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
}

// https://www.pcg-random.org/
UInt pcg(Expr<uint> v) noexcept {
    auto state = v * 747796405u + 2891336453u;
    auto word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

UInt2 pcg2d(Expr<uint2> v_in) noexcept {
    auto v = def(v_in);
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

// http://www.jcgt.org/published/0009/03/02/
UInt3 pcg3d(Expr<uint3> v_in) noexcept {
    auto v = def(v_in);
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// http://www.jcgt.org/published/0009/03/02/
UInt4 pcg4d(Expr<uint4> v_in) noexcept {
    auto v = def(v_in);
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
}

}// namespace luisa::render

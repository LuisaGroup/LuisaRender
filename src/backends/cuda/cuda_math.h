//
// Created by Mike on 8/28/2020.
//

#pragma once

#include <cstdint>
#include <type_traits>

namespace luisa::cuda {

namespace detail {

template<typename T>
constexpr auto always_false = false;

template<typename T, uint32_t N, bool is_packed>
constexpr auto vector_alignment = is_packed ? sizeof(T) : (sizeof(T) * (N + (N & 1u)));

template<typename T, uint32_t>
struct VectorStorage {};

template<typename T>
struct VectorStorage<T, 2> {
    T x, y;
    constexpr VectorStorage() noexcept : x{}, y{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s} {}
    constexpr VectorStorage(T x, T y) noexcept : x{x}, y{y} {}
};

template<typename T>
struct VectorStorage<T, 3> {
    T x, y, z;
    constexpr VectorStorage() noexcept : x{}, y{}, z{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s}, z{s} {}
    constexpr VectorStorage(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
};

template<typename T>
struct VectorStorage<T, 4> {
    T x, y, z, w;
    constexpr VectorStorage() noexcept : x{}, y{}, z{}, w{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s}, z{s}, w{s} {}
    constexpr VectorStorage(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
};

}

template<typename T, uint32_t N, bool is_packed>
struct alignas(detail::vector_alignment<T, N, is_packed>) Vector : detail::VectorStorage<T, N> {
    
    constexpr Vector() noexcept : detail::VectorStorage<T, N>{static_cast<T>(0)} {}
    
    template<typename U>
    explicit constexpr Vector(U u) noexcept : detail::VectorStorage<T, N>{static_cast<T>(u)} {}

    template<
        typename... U,
        std::enable_if_t<sizeof...(U) == N, int> = 0>
    explicit constexpr Vector(U ...u) noexcept : detail::VectorStorage<T, N>{static_cast<T>(u)...} {}
};

template<typename T, uint32_t N>
constexpr Vector<T, N, false> operator+(Vector<T, N, false> lhs, Vector<T, N, false> rhs) noexcept {
    if constexpr (N == 2) { return Vector<T, 2, false>{lhs.x + rhs.x, lhs.y + rhs.y}; }
    else if constexpr (N == 3) { return Vector<T, 3, false>{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z}; }
    else if constexpr (N == 4) { return Vector<T, 4, false>{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w}; }
    else { static_assert(detail::always_false<T>); }
}

using uchar = uint8_t;
using ushort = uint16_t;
using uint = uint32_t;

using char2 = Vector<char, 2, false>;
using char3 = Vector<char, 3, false>;
using char4 = Vector<char, 4, false>;
using packed_char3 = Vector<char, 3, true>;

using uchar2 = Vector<uchar, 2, false>;
using uchar3 = Vector<uchar, 3, false>;
using uchar4 = Vector<uchar, 4, false>;
using packed_uchar3 = Vector<uchar, 3, true>;

using short2 = Vector<short, 2, false>;
using short3 = Vector<short, 3, false>;
using short4 = Vector<short, 4, false>;
using packed_short3 = Vector<short, 3, true>;

using ushort2 = Vector<ushort, 2, false>;
using ushort3 = Vector<ushort, 3, false>;
using ushort4 = Vector<ushort, 4, false>;
using packed_ushort3 = Vector<ushort, 3, true>;

using int2 = Vector<int, 2, false>;
using int3 = Vector<int, 3, false>;
using int4 = Vector<int, 4, false>;
using packed_int3 = Vector<int, 3, true>;

using uint2 = Vector<uint, 2, false>;
using uint3 = Vector<uint, 3, false>;
using uint4 = Vector<uint, 4, false>;
using packed_uint3 = Vector<uint, 3, true>;

using float2 = Vector<float, 2, false>;
using float3 = Vector<float, 3, false>;
using float4 = Vector<float, 4, false>;
using packed_float3 = Vector<float, 3, true>;

static_assert(sizeof(uchar2) == 2u);
static_assert(sizeof(uchar3) == 4u);
static_assert(sizeof(uchar4) == 4u);
static_assert(sizeof(packed_uchar3) == 3u);

static_assert(alignof(uchar2) == 2u);
static_assert(alignof(uchar3) == 4u);
static_assert(alignof(uchar4) == 4u);
static_assert(alignof(packed_uchar3) == 1u);

static_assert(sizeof(float) == 4ul);
static_assert(sizeof(float2) == 8ul);
static_assert(sizeof(float3) == 16ul);
static_assert(sizeof(float4) == 16ul);
static_assert(sizeof(packed_float3) == 12ul);

static_assert(alignof(float) == 4ul);
static_assert(alignof(float2) == 8ul);
static_assert(alignof(float3) == 16ul);
static_assert(alignof(float4) == 16ul);
static_assert(alignof(packed_float3) == 4ul);

static_assert(sizeof(int) == 4ul);
static_assert(sizeof(int2) == 8ul);
static_assert(sizeof(int3) == 16ul);
static_assert(sizeof(int4) == 16ul);
static_assert(sizeof(packed_int3) == 12ul);

static_assert(alignof(int) == 4ul);
static_assert(alignof(int2) == 8ul);
static_assert(alignof(int3) == 16ul);
static_assert(alignof(int4) == 16ul);
static_assert(alignof(packed_int3) == 4ul);

static_assert(sizeof(uint) == 4ul);
static_assert(sizeof(uint2) == 8ul);
static_assert(sizeof(uint3) == 16ul);
static_assert(sizeof(uint4) == 16ul);
static_assert(sizeof(packed_uint3) == 12ul);

static_assert(alignof(uint) == 4ul);
static_assert(alignof(uint2) == 8ul);
static_assert(alignof(uint3) == 16ul);
static_assert(alignof(uint4) == 16ul);
static_assert(alignof(packed_uint3) == 4ul);

}// namespace luisa::cuda

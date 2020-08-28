//
// Created by Mike on 8/28/2020.
//

#pragma once

#include "scalar_types.h"
#include <type_traits>

namespace luisa {

namespace vec::detail {

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

}// namespace detail

template<typename T, uint32_t N, bool is_packed>
struct alignas(vec::detail::vector_alignment<T, N, is_packed>) Vector : vec::detail::VectorStorage<T, N> {

    constexpr Vector() noexcept : vec::detail::VectorStorage<T, N>{static_cast<T>(0)} {}

    template<typename U>
    explicit constexpr Vector(U u) noexcept : vec::detail::VectorStorage<T, N>{static_cast<T>(u)} {}

    template<
        typename... U,
        std::enable_if_t<sizeof...(U) == N, int> = 0>
    explicit constexpr Vector(U... u) noexcept : vec::detail::VectorStorage<T, N>{static_cast<T>(u)...} {}

    template<typename Index>
    [[nodiscard]] T &operator[](Index i) noexcept { return reinterpret_cast<T(&)[N]>(*this)[i]; }

    template<typename Index>
    [[nodiscard]] T operator[](Index i) const noexcept { return reinterpret_cast<const T(&)[N]>(*this)[i]; }
};

#define MAKE_VECTOR_BINARY_OP(op)                                                                          \
    template<typename T, uint32_t N>                                                                       \
    constexpr Vector<T, N, false> operator op(Vector<T, N, false> lhs, Vector<T, N, false> rhs) noexcept { \
        static_assert(N == 2 || N == 3 || N == 4);                                                         \
        if constexpr (N == 2) {                                                                            \
            return Vector<T, 2, false>{lhs.x op rhs.x, lhs.y op rhs.y};                                    \
        } else if constexpr (N == 3) {                                                                     \
            return Vector<T, 3, false>{lhs.x op rhs.x, lhs.y op rhs.y, lhs.z op rhs.z};                    \
        } else {                                                                                           \
            return Vector<T, 4, false>{lhs.x op rhs.x, lhs.y op rhs.y, lhs.z op rhs.z, lhs.w op rhs.w};    \
        }                                                                                                  \
    }                                                                                                      \
                                                                                                           \
    template<typename T, uint32_t N>                                                                       \
    constexpr Vector<T, N, false> operator op(T lhs, Vector<T, N, false> rhs) noexcept {                   \
        static_assert(N == 2 || N == 3 || N == 4);                                                         \
        if constexpr (N == 2) {                                                                            \
            return Vector<T, 2, false>{lhs op rhs.x, lhs op rhs.y};                                        \
        } else if constexpr (N == 3) {                                                                     \
            return Vector<T, 3, false>{lhs op rhs.x, lhs op rhs.y, lhs op rhs.z};                          \
        } else {                                                                                           \
            return Vector<T, 4, false>{lhs op rhs.x, lhs op rhs.y, lhs op rhs.z, lhs op rhs.w};            \
        }                                                                                                  \
    }                                                                                                      \
                                                                                                           \
    template<typename T, uint32_t N>                                                                       \
    constexpr Vector<T, N, false> operator op(Vector<T, N, false> lhs, T rhs) noexcept {                   \
        static_assert(N == 2 || N == 3 || N == 4);                                                         \
        if constexpr (N == 2) {                                                                            \
            return Vector<T, 2, false>{lhs.x op rhs, lhs.y op rhs};                                        \
        } else if constexpr (N == 3) {                                                                     \
            return Vector<T, 3, false>{lhs.x op rhs, lhs.y op rhs, lhs.z op rhs};                          \
        } else {                                                                                           \
            return Vector<T, 4, false>{lhs.x op rhs, lhs.y op rhs, lhs.z op rhs, lhs.w op rhs};            \
        }                                                                                                  \
    }

MAKE_VECTOR_BINARY_OP(+)
MAKE_VECTOR_BINARY_OP(-)
MAKE_VECTOR_BINARY_OP(*)
MAKE_VECTOR_BINARY_OP(/)
MAKE_VECTOR_BINARY_OP(%)

#undef MAKE_VECTOR_BINARY_OP

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

constexpr float2 make_float2() noexcept { return float2{}; }
constexpr float2 make_float2(float s) noexcept { return float2{s}; }
constexpr float2 make_float2(float x, float y) noexcept { return float2{x, y}; }
constexpr float2 make_float2(float3 v) noexcept { return float2{v.x, v.y}; }
constexpr float2 make_float2(float4 v) noexcept { return float2{v.x, v.y}; }

constexpr float3 make_float3() noexcept { return float3{}; }
constexpr float3 make_float3(float s) noexcept { return float3{s}; }
constexpr float3 make_float3(float x, float y, float z) noexcept { return float3{x, y, z}; }
constexpr float3 make_float3(float2 v, float z) noexcept { return float3{v.x, v.y, z}; }
constexpr float3 make_float3(float x, float2 v) noexcept { return float3{x, v.x, v.y}; }
constexpr float3 make_float3(float4 v) noexcept { return float3{v.x, v.y, v.z}; }

constexpr float4 make_float4() noexcept { return float4{}; }
constexpr float4 make_float4(float s) noexcept { return float4{s}; }
constexpr float4 make_float4(float x, float y, float z, float w) noexcept { return float4{x, y, z, w}; }
constexpr float4 make_float4(float2 v, float z, float w) noexcept { return float4{v.x, v.y, z, w}; }
constexpr float4 make_float4(float x, float y, float2 v) noexcept { return float4{x, y, v.x, v.y}; }
constexpr float4 make_float4(float x, float2 v, float w) noexcept { return float4{x, v.x, v.y, w}; }
constexpr float4 make_float4(float2 v, float2 u) noexcept { return float4{v.x, v.y, u.x, u.y}; }
constexpr float4 make_float4(float3 v, float w) noexcept { return float4{v.x, v.y, v.z, w}; }
constexpr float4 make_float4(float x, float3 v) noexcept { return float4{x, v.x, v.y, v.z}; }

constexpr int2 make_int2() noexcept { return int2{}; }
constexpr int2 make_int2(int s) noexcept { return int2{s}; }
constexpr int2 make_int2(int x, int y) noexcept { return int2{x, y}; }
constexpr int2 make_int2(int3 v) noexcept { return int2{v.x, v.y}; }
constexpr int2 make_int2(int4 v) noexcept { return int2{v.x, v.y}; }

constexpr int3 make_int3() noexcept { return int3{}; }
constexpr int3 make_int3(int s) noexcept { return int3{s}; }
constexpr int3 make_int3(int x, int y, int z) noexcept { return int3{x, y, z}; }
constexpr int3 make_int3(int2 v, int z) noexcept { return int3{v.x, v.y, z}; }
constexpr int3 make_int3(int x, int2 v) noexcept { return int3{x, v.x, v.y}; }
constexpr int3 make_int3(int4 v) noexcept { return int3{v.x, v.y, v.z}; }

constexpr int4 make_int4() noexcept { return int4{}; }
constexpr int4 make_int4(int s) noexcept { return int4{s}; }
constexpr int4 make_int4(int x, int y, int z, int w) noexcept { return int4{x, y, z, w}; }
constexpr int4 make_int4(int2 v, int z, int w) noexcept { return int4{v.x, v.y, z, w}; }
constexpr int4 make_int4(int x, int y, int2 v) noexcept { return int4{x, y, v.x, v.y}; }
constexpr int4 make_int4(int x, int2 v, int w) noexcept { return int4{x, v.x, v.y, w}; }
constexpr int4 make_int4(int2 v, int2 u) noexcept { return int4{v.x, v.y, u.x, u.y}; }
constexpr int4 make_int4(int3 v, int w) noexcept { return int4{v.x, v.y, v.z, w}; }
constexpr int4 make_int4(int x, int3 v) noexcept { return int4{x, v.x, v.y, v.z}; }

constexpr uint2 make_uint2() noexcept { return uint2{}; }
constexpr uint2 make_uint2(uint s) noexcept { return uint2{s}; }
constexpr uint2 make_uint2(uint x, uint y) noexcept { return uint2{x, y}; }
constexpr uint2 make_uint2(uint3 v) noexcept { return uint2{v.x, v.y}; }
constexpr uint2 make_uint2(uint4 v) noexcept { return uint2{v.x, v.y}; }

constexpr uint3 make_uint3() noexcept { return uint3{}; }
constexpr uint3 make_uint3(uint s) noexcept { return uint3{s}; }
constexpr uint3 make_uint3(uint x, uint y, uint z) noexcept { return uint3{x, y, z}; }
constexpr uint3 make_uint3(uint2 v, uint z) noexcept { return uint3{v.x, v.y, z}; }
constexpr uint3 make_uint3(uint x, uint2 v) noexcept { return uint3{x, v.x, v.y}; }
constexpr uint3 make_uint3(uint4 v) noexcept { return uint3{v.x, v.y, v.z}; }

constexpr uint4 make_uint4() noexcept { return uint4{}; }
constexpr uint4 make_uint4(uint s) noexcept { return uint4{s}; }
constexpr uint4 make_uint4(uint x, uint y, uint z, uint w) noexcept { return uint4{x, y, z, w}; }
constexpr uint4 make_uint4(uint2 v, uint z, uint w) noexcept { return uint4{v.x, v.y, z, w}; }
constexpr uint4 make_uint4(uint x, uint y, uint2 v) noexcept { return uint4{x, y, v.x, v.y}; }
constexpr uint4 make_uint4(uint x, uint2 v, uint w) noexcept { return uint4{x, v.x, v.y, w}; }
constexpr uint4 make_uint4(uint2 v, uint2 u) noexcept { return uint4{v.x, v.y, u.x, u.y}; }
constexpr uint4 make_uint4(uint3 v, uint w) noexcept { return uint4{v.x, v.y, v.z, w}; }
constexpr uint4 make_uint4(uint x, uint3 v) noexcept { return uint4{x, v.x, v.y, v.z}; }

constexpr float2 make_float2(int2 v) noexcept { return float2{static_cast<float>(v.x), static_cast<float>(v.y)}; }
constexpr float2 make_float2(uint2 v) noexcept { return float2{static_cast<float>(v.x), static_cast<float>(v.y)}; }
constexpr float3 make_float3(int3 v) noexcept { return float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }
constexpr float3 make_float3(uint3 v) noexcept { return float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }
constexpr float4 make_float4(int4 v) noexcept { return float4{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w)}; }
constexpr float4 make_float4(uint4 v) noexcept { return float4{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w)}; }
constexpr float3 make_float3(packed_int3 v) noexcept { return float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }
constexpr float3 make_float3(packed_uint3 v) noexcept { return float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }
constexpr float3 make_float3(packed_float3 v) noexcept { return float3{v.x, v.y, v.z}; }

constexpr int2 make_int2(float2 v) noexcept { return int2{static_cast<int>(v.x), static_cast<int>(v.y)}; }
constexpr int2 make_int2(uint2 v) noexcept { return int2{static_cast<int>(v.x), static_cast<int>(v.y)}; }
constexpr int3 make_int3(float3 v) noexcept { return int3{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)}; }
constexpr int3 make_int3(uint3 v) noexcept { return int3{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)}; }
constexpr int4 make_int4(float4 v) noexcept { return int4{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z), static_cast<int>(v.w)}; }
constexpr int4 make_int4(uint4 v) noexcept { return int4{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z), static_cast<int>(v.w)}; }
constexpr int3 make_int3(packed_int3 v) noexcept { return int3{v.x, v.y, v.z}; }
constexpr int3 make_int3(packed_uint3 v) noexcept { return int3{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)}; }
constexpr int3 make_int3(packed_float3 v) noexcept { return int3{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)}; }

constexpr uint2 make_uint2(float2 v) noexcept { return uint2{static_cast<uint>(v.x), static_cast<uint>(v.y)}; }
constexpr uint2 make_uint2(int2 v) noexcept { return uint2{static_cast<uint>(v.x), static_cast<uint>(v.y)}; }
constexpr uint3 make_uint3(float3 v) noexcept { return uint3{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z)}; }
constexpr uint3 make_uint3(int3 v) noexcept { return uint3{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z)}; }
constexpr uint4 make_uint4(float4 v) noexcept { return uint4{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z), static_cast<uint>(v.w)}; }
constexpr uint4 make_uint4(int4 v) noexcept { return uint4{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z), static_cast<uint>(v.w)}; }
constexpr uint3 make_uint3(packed_int3 v) noexcept { return uint3{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z)}; }
constexpr uint3 make_uint3(packed_uint3 v) noexcept { return uint3{v.x, v.y, v.z}; }
constexpr uint3 make_uint3(packed_float3 v) noexcept { return uint3{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z)}; }

constexpr packed_float3 make_packed_float3() noexcept { return packed_float3{}; }
constexpr packed_float3 make_packed_float3(int3 v) noexcept { return packed_float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }
constexpr packed_float3 make_packed_float3(uint3 v) noexcept { return packed_float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }
constexpr packed_float3 make_packed_float3(float3 v) noexcept { return packed_float3{v.x, v.y, v.z}; }
constexpr packed_float3 make_packed_float3(packed_int3 v) noexcept { return packed_float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }
constexpr packed_float3 make_packed_float3(packed_uint3 v) noexcept { return packed_float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)}; }

constexpr packed_int3 make_packed_int3(int3 v) noexcept { return packed_int3{v.x, v.y, v.z}; }
constexpr packed_int3 make_packed_int3(uint3 v) noexcept { return packed_int3{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)}; }
constexpr packed_int3 make_packed_int3(float3 v) noexcept { return packed_int3{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)}; }
constexpr packed_int3 make_packed_int3(packed_uint3 v) noexcept { return packed_int3{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)}; }
constexpr packed_int3 make_packed_int3(packed_float3 v) noexcept { return packed_int3{static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)}; }

constexpr packed_uint3 make_packed_uint3(uint3 v) noexcept { return packed_uint3{v.x, v.y, v.z}; }
constexpr packed_uint3 make_packed_uint3(int3 v) noexcept { return packed_uint3{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z)}; }
constexpr packed_uint3 make_packed_uint3(float3 v) noexcept { return packed_uint3{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z)}; }
constexpr packed_uint3 make_packed_uint3(packed_int3 v) noexcept { return packed_uint3{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z)}; }
constexpr packed_uint3 make_packed_uint3(packed_float3 v) noexcept { return packed_uint3{static_cast<uint>(v.x), static_cast<uint>(v.y), static_cast<uint>(v.z)}; }

}// namespace luisa

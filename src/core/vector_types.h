//
// Created by Mike on 8/28/2020.
//

#pragma once

#include "scalar_types.h"
#include <type_traits>

namespace luisa {

inline namespace vector {

namespace detail {

template<typename T, uint32_t N, bool is_packed>
constexpr auto vector_alignment = is_packed ? sizeof(T) : (sizeof(T) * (N + (N & 1u)));

template<typename T, uint32_t>
struct VectorStorage {};

template<typename T>
struct VectorStorage<T, 2> {
    T x, y;
    constexpr VectorStorage() noexcept: x{}, y{} {}
    explicit constexpr VectorStorage(T s) noexcept: x{s}, y{s} {}
    constexpr VectorStorage(T x, T y) noexcept: x{x}, y{y} {}
};

template<typename T>
struct VectorStorage<T, 3> {
    T x, y, z;
    constexpr VectorStorage() noexcept: x{}, y{}, z{} {}
    explicit constexpr VectorStorage(T s) noexcept: x{s}, y{s}, z{s} {}
    constexpr VectorStorage(T x, T y, T z) noexcept: x{x}, y{y}, z{z} {}
};

template<typename T>
struct VectorStorage<T, 4> {
    T x, y, z, w;
    constexpr VectorStorage() noexcept: x{}, y{}, z{}, w{} {}
    explicit constexpr VectorStorage(T s) noexcept: x{s}, y{s}, z{s}, w{s} {}
    constexpr VectorStorage(T x, T y, T z, T w) noexcept: x{x}, y{y}, z{z}, w{w} {}
};

}// namespace detail

template<typename T, uint32_t N, bool is_packed>
struct alignas(detail::vector_alignment<T, N, is_packed>) Vector : detail::VectorStorage<T, N> {
    
    using Storage = detail::VectorStorage<T, N>;
    
    constexpr Vector() noexcept: detail::VectorStorage<T, N>{static_cast<T>(0)} {}
    
    template<typename U>
    explicit constexpr Vector(U u) noexcept : detail::VectorStorage<T, N>{static_cast<T>(u)} {}
    
    template<
        typename... U,
        std::enable_if_t<sizeof...(U) == N, int> = 0>
    explicit constexpr Vector(U... u) noexcept : detail::VectorStorage<T, N>{static_cast<T>(u)...} {}
    
    template<typename Index>
    [[nodiscard]] T &operator[](Index i) noexcept { return reinterpret_cast<T(&)[N]>(*this)[i]; }
    
    template<typename Index>
    [[nodiscard]] T operator[](Index i) const noexcept { return reinterpret_cast<const T(&)[N]>(*this)[i]; }

#define MAKE_ASSIGN_OP(op)                                                 \
    template<bool packed>                                                  \
    [[nodiscard]] Vector &operator op(Vector<T, N, packed> rhs) noexcept { \
        static_assert(N == 2 || N == 3 || N == 4);                         \
        if constexpr (N == 2) {                                            \
            Storage::x op rhs.x;                                           \
            Storage::y op rhs.y;                                           \
        } else if constexpr (N == 3) {                                     \
            Storage::x op rhs.x;                                           \
            Storage::y op rhs.y;                                           \
            Storage::z op rhs.z;                                           \
        } else {                                                           \
            Storage::x op rhs.x;                                           \
            Storage::y op rhs.y;                                           \
            Storage::z op rhs.z;                                           \
            Storage::w op rhs.w;                                           \
        }                                                                  \
        return *this;                                                      \
    }                                                                      \
    [[nodiscard]] Vector &operator op(T rhs) noexcept {                    \
        static_assert(N == 2 || N == 3 || N == 4);                         \
        if constexpr (N == 2) {                                            \
            Storage::x op rhs;                                             \
            Storage::y op rhs;                                             \
        } else if constexpr (N == 3) {                                     \
            Storage::x op rhs;                                             \
            Storage::y op rhs;                                             \
            Storage::z op rhs;                                             \
        } else {                                                           \
            Storage::x op rhs;                                             \
            Storage::y op rhs;                                             \
            Storage::z op rhs;                                             \
            Storage::w op rhs;                                             \
        }                                                                  \
        return *this;                                                      \
    }
    
    MAKE_ASSIGN_OP(+=)
    MAKE_ASSIGN_OP(-=)
    MAKE_ASSIGN_OP(*=)
    MAKE_ASSIGN_OP(/=)
    MAKE_ASSIGN_OP(%=)

#undef MAKE_ASSIGN_OP
};

#define MAKE_VECTOR_BINARY_OP(op)                                                                                        \
    template<typename T, uint32_t N, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>                                 \
    [[nodiscard]] constexpr Vector<T, N, false> operator op(Vector<T, N, false> lhs, Vector<T, N, false> rhs) noexcept { \
        static_assert(N == 2 || N == 3 || N == 4);                                                                       \
        if constexpr (N == 2) {                                                                                          \
            return Vector<T, 2, false>{lhs.x op rhs.x, lhs.y op rhs.y};                                                  \
        } else if constexpr (N == 3) {                                                                                   \
            return Vector<T, 3, false>{lhs.x op rhs.x, lhs.y op rhs.y, lhs.z op rhs.z};                                  \
        } else {                                                                                                         \
            return Vector<T, 4, false>{lhs.x op rhs.x, lhs.y op rhs.y, lhs.z op rhs.z, lhs.w op rhs.w};                  \
        }                                                                                                                \
    }                                                                                                                    \
                                                                                                                         \
    template<typename T, uint32_t N, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>                                 \
    [[nodiscard]] constexpr Vector<T, N, false> operator op(T lhs, Vector<T, N, false> rhs) noexcept {                   \
        static_assert(N == 2 || N == 3 || N == 4);                                                                       \
        if constexpr (N == 2) {                                                                                          \
            return Vector<T, 2, false>{lhs op rhs.x, lhs op rhs.y};                                                      \
        } else if constexpr (N == 3) {                                                                                   \
            return Vector<T, 3, false>{lhs op rhs.x, lhs op rhs.y, lhs op rhs.z};                                        \
        } else {                                                                                                         \
            return Vector<T, 4, false>{lhs op rhs.x, lhs op rhs.y, lhs op rhs.z, lhs op rhs.w};                          \
        }                                                                                                                \
    }                                                                                                                    \
                                                                                                                         \
    template<typename T, uint32_t N, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>                                 \
    [[nodiscard]] constexpr Vector<T, N, false> operator op(Vector<T, N, false> lhs, T rhs) noexcept {                   \
        static_assert(N == 2 || N == 3 || N == 4);                                                                       \
        if constexpr (N == 2) {                                                                                          \
            return Vector<T, 2, false>{lhs.x op rhs, lhs.y op rhs};                                                      \
        } else if constexpr (N == 3) {                                                                                   \
            return Vector<T, 3, false>{lhs.x op rhs, lhs.y op rhs, lhs.z op rhs};                                        \
        } else {                                                                                                         \
            return Vector<T, 4, false>{lhs.x op rhs, lhs.y op rhs, lhs.z op rhs, lhs.w op rhs};                          \
        }                                                                                                                \
    }

MAKE_VECTOR_BINARY_OP(+)
MAKE_VECTOR_BINARY_OP(-)
MAKE_VECTOR_BINARY_OP(*)
MAKE_VECTOR_BINARY_OP(/)
MAKE_VECTOR_BINARY_OP(%)

#undef MAKE_VECTOR_BINARY_OP

#define MAKE_VECTOR_RELATIONAL_OP(op)                                                                                    \
    template<typename T, uint N>                                                                                         \
    [[nodiscard]] constexpr auto operator op(Vector<T, N, false> lhs, Vector<T, N, false> rhs) noexcept {                \
        static_assert(N == 2 || N == 3 || N == 4);                                                                       \
        if constexpr (N == 2) {                                                                                          \
            return Vector<bool, 2, false>{lhs.x op rhs.x, lhs.y op rhs.y};                                               \
        } else if constexpr (N == 3) {                                                                                   \
            return Vector<bool, 3, false>{lhs.x op rhs.x, lhs.y op rhs.y, lhs.z op rhs.z};                               \
        } else {                                                                                                         \
            return Vector<bool, 4, false>{lhs.x op rhs.x, lhs.y op rhs.y, lhs.z op rhs.z, lhs.w op rhs.w};               \
        }                                                                                                                \
    }                                                                                                                    \
                                                                                                                         \
    template<typename T, uint N>                                                                                         \
    [[nodiscard]] constexpr auto operator op(T lhs, Vector<T, N, false> rhs) noexcept {                                  \
        static_assert(N == 2 || N == 3 || N == 4);                                                                       \
        if constexpr (N == 2) {                                                                                          \
            return Vector<bool, 2, false>{lhs op rhs.x, lhs op rhs.y};                                                   \
        } else if constexpr (N == 3) {                                                                                   \
            return Vector<bool, 3, false>{lhs op rhs.x, lhs op rhs.y, lhs op rhs.z};                                     \
        } else {                                                                                                         \
            return Vector<bool, 4, false>{lhs op rhs.x, lhs op rhs.y, lhs op rhs.z, lhs op rhs.w};                       \
        }                                                                                                                \
    }                                                                                                                    \
                                                                                                                         \
    template<typename T, uint N>                                                                                         \
    [[nodiscard]] constexpr auto operator op(Vector<T, N, false> lhs, T rhs) noexcept {                                  \
        static_assert(N == 2 || N == 3 || N == 4);                                                                       \
        if constexpr (N == 2) {                                                                                          \
            return Vector<bool, 2, false>{lhs.x op rhs, lhs.y op rhs};                                                   \
        } else if constexpr (N == 3) {                                                                                   \
            return Vector<bool, 3, false>{lhs.x op rhs, lhs.y op rhs, lhs.z op rhs};                                     \
        } else {                                                                                                         \
            return Vector<bool, 4, false>{lhs.x op rhs, lhs.y op rhs, lhs.z op rhs, lhs.w op rhs};                       \
        }                                                                                                                \
    }

MAKE_VECTOR_RELATIONAL_OP(==)
MAKE_VECTOR_RELATIONAL_OP(!=)
MAKE_VECTOR_RELATIONAL_OP(<)
MAKE_VECTOR_RELATIONAL_OP(<=)
MAKE_VECTOR_RELATIONAL_OP(>)
MAKE_VECTOR_RELATIONAL_OP(>=)

#undef MAKE_VECTOR_RELATIONAL_OP

#define MAKE_VECTOR_MAKE_TYPE2(type)                                                                \
    [[nodiscard]] constexpr auto make_##type##2() noexcept { return type##2 {}; }                   \
    [[nodiscard]] constexpr auto make_##type##2(type s) noexcept { return type##2 {s}; }            \
    [[nodiscard]] constexpr auto make_##type##2(type x, type y) noexcept { return type##2 {x, y}; } \
                                                                                                    \
    template<typename U, uint N, bool packed>                                                       \
    [[nodiscard]] constexpr auto make_##type##2(Vector<U, N, packed> v) noexcept {                  \
        static_assert(N == 2 || N == 3 || N == 4);                                                  \
        return type##2 {v.x, v.y};                                                                  \
    }

#define MAKE_VECTOR_MAKE_TYPE3(type)                                                                                           \
    [[nodiscard]] constexpr auto make_##type##3() noexcept { return type##3 {}; }                                              \
    [[nodiscard]] constexpr auto make_##type##3(type s) noexcept { return type##3 {s}; }                                       \
    [[nodiscard]] constexpr auto make_##type##3(type x, type y, type z) noexcept { return type##3 {x, y, z}; }                 \
    [[nodiscard]] constexpr auto make_##type##3(type##2 v, type z) noexcept { return type##3 {v.x, v.y, z}; }                  \
    [[nodiscard]] constexpr auto make_##type##3(type x, type##2 v) noexcept { return type##3 {x, v.x, v.y}; }                  \
                                                                                                                               \
    [[nodiscard]] constexpr auto make_packed_##type##3() noexcept { return packed_##type##3 {}; }                              \
    [[nodiscard]] constexpr auto make_packed_##type##3(type s) noexcept { return packed_##type##3 {s}; }                       \
    [[nodiscard]] constexpr auto make_packed_##type##3(type x, type y, type z) noexcept { return packed_##type##3 {x, y, z}; } \
    [[nodiscard]] constexpr auto make_packed_##type##3(type##2 v, type z) noexcept { return packed_##type##3 {v.x, v.y, z}; }  \
    [[nodiscard]] constexpr auto make_packed_##type##3(type x, type##2 v) noexcept { return packed_##type##3 {x, v.x, v.y}; }  \
                                                                                                                               \
    template<typename U, uint N, bool packed>                                                                                  \
    [[nodiscard]] constexpr auto make_##type##3(Vector<U, N, packed> v) noexcept {                                             \
        static_assert(N == 3 || N == 4);                                                                                       \
        return type##3 {v.x, v.y, v.z};                                                                                        \
    }                                                                                                                          \
                                                                                                                               \
    template<typename U, uint N, bool packed>                                                                                  \
    [[nodiscard]] constexpr auto make_packed_##type##3(Vector<U, N, packed> v) noexcept {                                      \
        static_assert(N == 3 || N == 4);                                                                                       \
        return packed_##type##3 {v.x, v.y, v.z};                                                                               \
    }

#define MAKE_VECTOR_MAKE_TYPE4(type)                                                                                        \
    [[nodiscard]] constexpr auto make_##type##4() noexcept { return type##4 {}; }                                           \
    [[nodiscard]] constexpr auto make_##type##4(type s) noexcept { return type##4 {s}; }                                    \
    [[nodiscard]] constexpr auto make_##type##4(type x, type y, type z, type w) noexcept { return type##4 {x, y, z, w}; }   \
    [[nodiscard]] constexpr auto make_##type##4(type##2 v, type z, type w) noexcept { return type##4 {v.x, v.y, z, w}; }    \
    [[nodiscard]] constexpr auto make_##type##4(type x, type y, type##2 v) noexcept { return type##4 {x, y, v.x, v.y}; }    \
    [[nodiscard]] constexpr auto make_##type##4(type x, type##2 v, type w) noexcept { return type##4 {x, v.x, v.y, w}; }    \
    [[nodiscard]] constexpr auto make_##type##4(type##2 v, type##2 u) noexcept { return type##4 {v.x, v.y, u.x, u.y}; }     \
    [[nodiscard]] constexpr auto make_##type##4(type##3 v, type w) noexcept { return type##4 {v.x, v.y, v.z, w}; }          \
    [[nodiscard]] constexpr auto make_##type##4(type x, type##3 v) noexcept { return type##4 {x, v.x, v.y, v.z}; }          \
    [[nodiscard]] constexpr auto make_##type##4(packed_##type##3 v, type w) noexcept { return type##4 {v.x, v.y, v.z, w}; } \
    [[nodiscard]] constexpr auto make_##type##4(type x, packed_##type##3 v) noexcept { return type##4 {x, v.x, v.y, v.z}; } \
    template<typename U>                                                                                                    \
    [[nodiscard]] constexpr auto make_##type##4(Vector<U, 4, false> v) noexcept { return type##4 {v.x, v.y, v.z, v.w}; }

#define MAKE_VECTOR_TYPE(type)                      \
    using type##2 = Vector<type, 2, false>;         \
    using type##3 = Vector<type, 3, false>;         \
    using type##4 = Vector<type, 4, false>;         \
    using packed_##type##3 = Vector<type, 3, true>; \
    MAKE_VECTOR_MAKE_TYPE2(type)                    \
    MAKE_VECTOR_MAKE_TYPE3(type)                    \
    MAKE_VECTOR_MAKE_TYPE4(type)

MAKE_VECTOR_TYPE(bool)
MAKE_VECTOR_TYPE(char)
MAKE_VECTOR_TYPE(uchar)
MAKE_VECTOR_TYPE(short)
MAKE_VECTOR_TYPE(ushort)
MAKE_VECTOR_TYPE(int)
MAKE_VECTOR_TYPE(uint)
MAKE_VECTOR_TYPE(float)

#undef MAKE_VECTOR_TYPE
#undef MAKE_VECTOR_MAKE_TYPE2
#undef MAKE_VECTOR_MAKE_TYPE3
#undef MAKE_VECTOR_MAKE_TYPE4

// For boolN
[[nodiscard]] constexpr auto operator!(bool2 v) noexcept { return make_bool2(!v.x, !v.y); }
[[nodiscard]] constexpr auto operator!(bool3 v) noexcept { return make_bool3(!v.x, !v.y, !v.z); }
[[nodiscard]] constexpr auto operator!(bool4 v) noexcept { return make_bool4(!v.x, !v.y, !v.z, !v.w); }
[[nodiscard]] constexpr auto operator||(bool2 lhs, bool2 rhs) noexcept { return make_bool2(lhs.x || rhs.x, lhs.y || rhs.y); }
[[nodiscard]] constexpr auto operator||(bool3 lhs, bool3 rhs) noexcept { return make_bool3(lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z); }
[[nodiscard]] constexpr auto operator||(bool4 lhs, bool4 rhs) noexcept { return make_bool4(lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z, lhs.w || rhs.w); }
[[nodiscard]] constexpr auto operator&&(bool2 lhs, bool2 rhs) noexcept { return make_bool2(lhs.x && rhs.x, lhs.y && rhs.y); }
[[nodiscard]] constexpr auto operator&&(bool3 lhs, bool3 rhs) noexcept { return make_bool3(lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z); }
[[nodiscard]] constexpr auto operator&&(bool4 lhs, bool4 rhs) noexcept { return make_bool4(lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z, lhs.w && rhs.w); }

[[nodiscard]] constexpr bool any(bool2 v) noexcept { return v.x || v.y; }
[[nodiscard]] constexpr bool any(bool3 v) noexcept { return v.x || v.y || v.z; }
[[nodiscard]] constexpr bool any(bool4 v) noexcept { return v.x || v.y || v.z || v.w; }
[[nodiscard]] constexpr bool all(bool2 v) noexcept { return v.x && v.y; }
[[nodiscard]] constexpr bool all(bool3 v) noexcept { return v.x && v.y && v.z; }
[[nodiscard]] constexpr bool all(bool4 v) noexcept { return v.x && v.y && v.z && v.w; }
[[nodiscard]] constexpr bool none(bool2 v) noexcept { return !any(v); }
[[nodiscard]] constexpr bool none(bool3 v) noexcept { return !any(v); }
[[nodiscard]] constexpr bool none(bool4 v) noexcept { return !any(v); }

namespace detail {

template<typename T>
struct IsVector2Impl : std::false_type {};

template<typename T, bool packed>
struct IsVector2Impl<Vector<T, 2, packed>> : std::true_type {};

template<typename T>
struct IsVector3Impl : std::false_type {};

template<typename T, bool packed>
struct IsVector3Impl<Vector<T, 3, packed>> : std::true_type {};

template<typename T>
struct IsVector4Impl : std::false_type {};

template<typename T, bool packed>
struct IsVector4Impl<Vector<T, 3, packed>> : std::true_type {};

}

template<typename T> constexpr auto is_vector_2 = detail::IsVector2Impl<T>::value;
template<typename T> constexpr auto is_vector_3 = detail::IsVector3Impl<T>::value;
template<typename T> constexpr auto is_vector_4 = detail::IsVector4Impl<T>::value;
template<typename T> constexpr auto is_vector = std::disjunction_v<detail::IsVector2Impl<T>, detail::IsVector3Impl<T>, detail::IsVector4Impl<T>>;

}
}// namespace luisa::vector

//
// Created by Mike Smith on 2020/1/30.
//

#pragma once

#ifdef LUISA_DEVICE_COMPATIBLE

#include <metal_stdlib>

namespace luisa {

using namespace simd;

namespace math {

using metal::sqrt;

template<typename A, typename B>
auto lerp(A a, B b, float t) {
    return (1.0f - t) * a + t * b;
}

using metal::cos;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_COS

using metal::sin;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_SIN

using metal::tan;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_TAN

using metal::acos;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ACOS

using metal::asin;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ASIN

using metal::atan;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN

using metal::atan2;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN2

using metal::ceil;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_CEIL

using metal::floor;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_FLOOR

using metal::round;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ROUND

using metal::log;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_LOG

using metal::exp;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_EXP

using metal::log2;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_LOG2

using metal::log10;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_LOG10

using metal::pow;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_POW

using metal::min;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_MIN

using metal::max;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_MAX

using metal::abs;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ABS

using metal::clamp;
#define LUISA_MATH_HAS_BUILTIN_VECTOR_CLAMP

#define LUISA_MATH_HAS_BUILTIN_MATRIX_TRANSPOSE

using metal::cross;

}}

#define noexcept

#define LUISA_CONSTANT_SPACE  constant
#define LUISA_UNIFORM_SPACE   constant
#define LUISA_THREAD_SPACE    thread
#define LUISA_DEVICE_SPACE    device
#define LUISA_THREAD_ID_DECL  uint $tid$ [[thread_position_in_grid]]
#define LUISA_THREAD_ID       $tid$

// function scopes
#define LUISA_CONSTEXPR        inline
#define LUISA_KERNEL           kernel
#define LUISA_DEVICE_CALLABLE

namespace luisa {

template<typename D, typename S>
inline D as(S s) { return as_type<D>(s); }

#define LUISA_STD_ATOMIC_COMPATIBLE

namespace _impl {

using atomic_int = metal::atomic_int;
using atomic_uint = metal::atomic_uint;

using metal::atomic_load_explicit;
using metal::atomic_store_explicit;
using metal::atomic_exchange_explicit;
using metal::atomic_fetch_add_explicit;
using metal::atomic_fetch_sub_explicit;
using metal::atomic_fetch_or_explicit;
using metal::atomic_fetch_and_explicit;
using metal::atomic_fetch_xor_explicit;

using metal::memory_order_relaxed;

}

}

#define static_assert(pred) static_assert(pred, "")

#endif

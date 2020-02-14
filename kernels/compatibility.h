//
// Created by Mike Smith on 2020/1/30.
//

#pragma once

#ifdef __METAL_VERSION__

#define LUISA_DEVICE_COMPATIBLE

#include <metal_stdlib>

namespace luisa {

using namespace simd;

inline namespace math {

using namespace metal;

#define LUISA_MATH_HAS_BUILTIN_VECTOR_COS
#define LUISA_MATH_HAS_BUILTIN_VECTOR_SIN
#define LUISA_MATH_HAS_BUILTIN_VECTOR_TAN
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ACOS
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ASIN
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ATAN2
#define LUISA_MATH_HAS_BUILTIN_VECTOR_CEIL
#define LUISA_MATH_HAS_BUILTIN_VECTOR_FLOOR
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ROUND
#define LUISA_MATH_HAS_BUILTIN_VECTOR_LOG
#define LUISA_MATH_HAS_BUILTIN_VECTOR_EXP
#define LUISA_MATH_HAS_BUILTIN_VECTOR_LOG2
#define LUISA_MATH_HAS_BUILTIN_VECTOR_LOG10
#define LUISA_MATH_HAS_BUILTIN_VECTOR_POW
#define LUISA_MATH_HAS_BUILTIN_VECTOR_MIN
#define LUISA_MATH_HAS_BUILTIN_VECTOR_MAX
#define LUISA_MATH_HAS_BUILTIN_VECTOR_ABS
#define LUISA_MATH_HAS_BUILTIN_VECTOR_CLAMP

#define LUISA_MATH_HAS_BUILTIN_MATRIX_TRANSPOSE

}}

#define noexcept

#define LUISA_CONSTANT_SPACE  constant
#define LUISA_UNIFORM_SPACE   constant
#define LUISA_THREAD_SPACE    thread
#define LUISA_DEVICE_SPACE    device

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

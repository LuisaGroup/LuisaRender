//
// Created by Mike on 8/28/2020.
//

#pragma once

#include <cstdint>

namespace luisa {

inline namespace scalar {

using uchar = uint8_t;
using ushort = uint16_t;
using uint = uint32_t;

template<typename T>
struct IsScalar : std::false_type {};

#define MAKE_IS_SCALAR_TRUE(Type) template<> struct IsScalar<Type> : std::true_type {};

MAKE_IS_SCALAR_TRUE(bool)
MAKE_IS_SCALAR_TRUE(float)
MAKE_IS_SCALAR_TRUE(int8_t)
MAKE_IS_SCALAR_TRUE(uint8_t)
MAKE_IS_SCALAR_TRUE(int16_t)
MAKE_IS_SCALAR_TRUE(uint16_t)
MAKE_IS_SCALAR_TRUE(int32_t)
MAKE_IS_SCALAR_TRUE(uint32_t)
#undef MAKE_IS_SCALAR_TRUE

template<typename T>
constexpr auto is_scalar = IsScalar<T>::value;

}}

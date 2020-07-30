//
// Created by Mike Smith on 2020/7/30.
//

#pragma once

#include <compute/function.h>

namespace luisa::dsl {

#define MAP_VARIABLE_NAME_TO_ARGUMENT_DEF(name) Variable name
#define MAP_VARIABLE_NAMES_TO_ARGUMENT_LIST(...) LUISA_MAP_MACRO_LIST(MAP_VARIABLE_NAME_TO_ARGUMENT_DEF, __VA_ARGS__)

#define MAKE_BUILTIN_FUNCTION_DEF(func, func_tag, ...)                                                    \
inline Variable func(MAP_VARIABLE_NAMES_TO_ARGUMENT_LIST(__VA_ARGS__)) {                                  \
    std::vector<Variable> args{__VA_ARGS__};                                                              \
    auto f = args.front().function();                                                                     \
    return f->add_expression(std::make_unique<BuiltinFuncExpr>(BuiltinFunc::func_tag, std::move(args)));  \
}                                                                                                         \

MAKE_BUILTIN_FUNCTION_DEF(select_, SELECT, cond, tv, fv)

MAKE_BUILTIN_FUNCTION_DEF(sin_, SIN, x)
MAKE_BUILTIN_FUNCTION_DEF(cos_, COS, x)
MAKE_BUILTIN_FUNCTION_DEF(tan_, TAN, x)
MAKE_BUILTIN_FUNCTION_DEF(asin_, ASIN, x)
MAKE_BUILTIN_FUNCTION_DEF(acos_, ACOS, x)
MAKE_BUILTIN_FUNCTION_DEF(atan_, ATAN, x)
MAKE_BUILTIN_FUNCTION_DEF(atan2_, ATAN2, y, x)
MAKE_BUILTIN_FUNCTION_DEF(ceil_, CEIL, x)
MAKE_BUILTIN_FUNCTION_DEF(floor_, FLOOR, x)
MAKE_BUILTIN_FUNCTION_DEF(round_, ROUND, x)
MAKE_BUILTIN_FUNCTION_DEF(pow_, POW, x)
MAKE_BUILTIN_FUNCTION_DEF(exp_, EXP, x)
MAKE_BUILTIN_FUNCTION_DEF(log_, LOG, x)
MAKE_BUILTIN_FUNCTION_DEF(log2_, LOG2, x)
MAKE_BUILTIN_FUNCTION_DEF(log10_, LOG10, x)
MAKE_BUILTIN_FUNCTION_DEF(min_, MIN, x, y)
MAKE_BUILTIN_FUNCTION_DEF(max_, MAX, x, y)
MAKE_BUILTIN_FUNCTION_DEF(abs_, ABS, x)
MAKE_BUILTIN_FUNCTION_DEF(clamp_, CLAMP, x, a, b)
MAKE_BUILTIN_FUNCTION_DEF(lerp_, LERP, a, b, t)
MAKE_BUILTIN_FUNCTION_DEF(radians_, RADIANS, deg)
MAKE_BUILTIN_FUNCTION_DEF(degrees_, DEGREES, rad)
MAKE_BUILTIN_FUNCTION_DEF(normalize_, NORMALIZE, v)
MAKE_BUILTIN_FUNCTION_DEF(length_, LENGTH, v)
MAKE_BUILTIN_FUNCTION_DEF(dot_, DOT, u, v)
MAKE_BUILTIN_FUNCTION_DEF(cross_, CROSS, u, v)

MAKE_BUILTIN_FUNCTION_DEF(make_mat3_, MAKE_MAT3, val_or_mat4)
MAKE_BUILTIN_FUNCTION_DEF(make_mat3_, MAKE_MAT3, c0, c1, c2)
MAKE_BUILTIN_FUNCTION_DEF(make_mat3_, MAKE_MAT3, m00, m01, m02, m10, m11, m12, m20, m21, m22)

MAKE_BUILTIN_FUNCTION_DEF(make_mat4_, MAKE_MAT4, val_or_mat3)
MAKE_BUILTIN_FUNCTION_DEF(make_mat4_, MAKE_MAT4, c0, c1, c2, c3)
MAKE_BUILTIN_FUNCTION_DEF(make_mat4_, MAKE_MAT4, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33)

// make_vec2
#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC2(T, tag)                      \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##2_, MAKE_##tag##2, v)                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##2_, MAKE_##tag##2, x, y)             \

// make_vec3
#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC3(T, tag)                      \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##3_, MAKE_##tag##3, v)                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##3_, MAKE_##tag##3, x, y)             \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##3_, MAKE_##tag##3, x, y, z)          \

// make_vec4
#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC4(T, tag)                      \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##4_, MAKE_##tag##4, v)                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##4_, MAKE_##tag##4, x, y)             \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##4_, MAKE_##tag##4, x, y, z)          \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##4_, MAKE_##tag##4, x, y, z, w)       \

// make_packed_vec3
#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_PACKED_VEC3(T, tag)               \
MAKE_BUILTIN_FUNCTION_DEF(make_packed_##T##3_, MAKE_PACKED_##tag##3, v)  \

#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(T, tag)                       \
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC2(T, tag)                              \
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC3(T, tag)                              \
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC4(T, tag)                              \
MAKE_BUILTIN_FUNCTION_DEF_MAKE_PACKED_VEC3(T, tag)                       \

MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(bool, BOOL)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(float, FLOAT)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(byte, BYTE)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(ubyte, UBYTE)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(short, SHORT)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(ushort, USHORT)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(int, INT)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(uint, UINT)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(long, LONG)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(ulong, ULONG)

#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC
#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC4
#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC3
#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC2

MAKE_BUILTIN_FUNCTION_DEF(inverse, INVERSE, m)
MAKE_BUILTIN_FUNCTION_DEF(transpose, TRANSPOSE, m)

#undef MAKE_BUILTIN_FUNCTION_DEF
#undef MAP_VARIABLE_NAMES_TO_ARGUMENT_LIST
#undef MAP_VARIABLE_NAME_TO_ARGUMENT_DEF

template<typename T>
[[nodiscard]] inline Variable static_cast_(Variable v) {
    return v.function()->add_expression(std::make_unique<CastExpr>(CastOp::STATIC, v, type_desc<T>));
}

template<typename T>
[[nodiscard]] inline Variable reinterpret_cast_(Variable v) {
    return v.function()->add_expression(std::make_unique<CastExpr>(CastOp::REINTERPRET, v, type_desc<T>));
}

template<typename T>
[[nodiscard]] inline Variable bitwise_cast_(Variable v) {
    return v.function()->add_expression(std::make_unique<CastExpr>(CastOp::BITWISE, v, type_desc<T>));
}

}

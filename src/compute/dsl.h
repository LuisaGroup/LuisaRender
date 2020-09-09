//
// Created by Mike Smith on 2020/8/5.
//

#pragma once

#include <type_traits>

#include <core/math_util.h>
#include <core/atomic_util.h>
#include <compute/variable.h>
#include <compute/function.h>
#include <compute/expression.h>
#include <compute/statement.h>

namespace luisa::compute::dsl {

// Forward decls
class ExprBase;

template<typename T>
struct Expr;

template<typename T>
struct Var;

// Some traits
template<typename T>
using IsExpr = std::is_base_of<ExprBase, std::decay_t<T>>;

template<typename T>
constexpr auto is_expr = IsExpr<T>::value;

class ExprBase {

protected:
    const Variable *_variable{nullptr};

public:
    explicit ExprBase(const Variable *variable) noexcept: _variable{variable} {}
    [[nodiscard]] const Variable *variable() const noexcept { return _variable; }

#define MAKE_ASSIGN_OP_EXPR(op, op_tag)                                                 \
    void operator op(const ExprBase &rhs) const noexcept {                              \
        Function::current().add_statement(                                              \
            std::make_unique<AssignStmt>(AssignOp::op_tag, _variable, rhs._variable));  \
    }

#define MAKE_ASSIGN_OP_SCALAR(T, op, op_tag)                                              \
    void operator op(T x) const noexcept {                                                \
        auto v = Variable::make_temporary(type_desc<T>, std::make_unique<ValueExpr>(x));  \
        Function::current().add_statement(                                                \
            std::make_unique<AssignStmt>(AssignOp::op_tag, _variable, v));                \
    }

#define MAKE_ASSIGN_OP(op, op_tag)               \
    MAKE_ASSIGN_OP_EXPR(op, op_tag)              \
    MAKE_ASSIGN_OP_SCALAR(bool, op, op_tag)      \
    MAKE_ASSIGN_OP_SCALAR(float, op, op_tag)     \
    MAKE_ASSIGN_OP_SCALAR(int8_t, op, op_tag)    \
    MAKE_ASSIGN_OP_SCALAR(uint8_t, op, op_tag)   \
    MAKE_ASSIGN_OP_SCALAR(int16_t, op, op_tag)   \
    MAKE_ASSIGN_OP_SCALAR(uint16_t, op, op_tag)  \
    MAKE_ASSIGN_OP_SCALAR(int32_t, op, op_tag)   \
    MAKE_ASSIGN_OP_SCALAR(uint32_t, op, op_tag)
    
    MAKE_ASSIGN_OP(=, ASSIGN)
    MAKE_ASSIGN_OP(+=, ADD_ASSIGN)
    MAKE_ASSIGN_OP(-=, SUB_ASSIGN)
    MAKE_ASSIGN_OP(*=, MUL_ASSIGN)
    MAKE_ASSIGN_OP(/=, DIV_ASSIGN)
    MAKE_ASSIGN_OP(%=, MOD_ASSIGN)
    MAKE_ASSIGN_OP(<<=, SHL_ASSIGN)
    MAKE_ASSIGN_OP(>>=, SHR_ASSIGN)
    MAKE_ASSIGN_OP(|=, BIT_OR_ASSIGN)
    MAKE_ASSIGN_OP(^=, BIT_XOR_ASSIGN)
    MAKE_ASSIGN_OP(&=, BIT_AND_ASSIGN)

#undef MAKE_ASSIGN_OP
#undef MAKE_ASSIGN_OP_EXPR
#undef MAKE_ASSIGN_OP_SCALAR
};

template<typename T>
struct IsArray : std::false_type {};

template<typename T, size_t N>
struct IsArray<T[N]> : std::true_type {};

template<typename T, size_t N>
struct IsArray<std::array<T, N>> : std::true_type {};

template<typename T>
constexpr auto is_array = IsArray<T>::value;

template<typename T>
struct Expr : public ExprBase {
    
    explicit Expr(const Variable *v) noexcept: ExprBase{v} {}
    
    Expr(T v) noexcept: ExprBase{Variable::make_temporary(type_desc<T>, std::make_unique<ValueExpr>(v))} {}
    
    Expr(ExprBase &&expr) noexcept: ExprBase{expr.variable()} {}
    Expr(const ExprBase &expr) noexcept: ExprBase{expr.variable()} {}
    
    [[nodiscard]] auto x() const noexcept {
        if constexpr (is_vector<T>) {
            using MemberType = std::decay_t<decltype(std::declval<T>().x)>;
            return Expr<MemberType>(Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, "x")));
        } else { LUISA_ERROR("This variable has no component x."); }
    }
    
    [[nodiscard]] auto y() const noexcept {
        if constexpr (is_vector<T>) {
            using MemberType = std::decay_t<decltype(std::declval<T>().y)>;
            return Expr<MemberType>{Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, "y"))};
        } else { LUISA_ERROR("This variable has no component y."); }
    }
    
    [[nodiscard]] auto z() const noexcept {
        if constexpr (is_vector_3<T> || is_vector_4<T>) {
            using MemberType = std::decay_t<decltype(std::declval<T>().z)>;
            return Expr<MemberType>{Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, "z"))};
        } else { LUISA_ERROR("This variable has no component z."); }
    }
    
    [[nodiscard]] auto w() const noexcept {
        if constexpr (is_vector_4<T>) {
            using MemberType = std::decay_t<decltype(std::declval<T>().w)>;
            return Expr<MemberType>{Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, "w"))};
        } else { LUISA_ERROR("This variable has no component w."); }
    }
    
    [[nodiscard]] auto r() const noexcept { return x(); }
    [[nodiscard]] auto g() const noexcept { return y(); }
    [[nodiscard]] auto b() const noexcept { return z(); }
    [[nodiscard]] auto a() const noexcept { return w(); }
    
    [[nodiscard]] auto operator[](const ExprBase &index) const noexcept {
        if constexpr (is_array<std::decay_t<T>>) {
            using ElementType = std::decay_t<decltype(std::declval<T>()[0])>;
            return Expr<ElementType>{Variable::make_temporary(type_desc<ElementType>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, _variable, index.variable()))};
        } else { LUISA_ERROR("operator[] is not available on non-array types."); }
    }
};

// Deduction guides
template<typename T, std::enable_if_t<!is_expr<T>, int> = 0>
Expr(T &&) -> Expr<T>;

template<typename T>
Expr(const Var<T> &) -> Expr<T>;

template<typename T>
Expr(Var<T> &&) -> Expr<T>;

template<typename T>
Expr(const Expr<T> &) -> Expr<T>;

template<typename T>
Expr(Expr<T> &&) -> Expr<T>;

namespace detail {

template<typename T>
inline const Variable *extract_variable(T &&v) noexcept {
    Expr v_expr{v};
    return v_expr.variable();
}

}

template<typename T>
struct Var : public Expr<T> {
    
    template<typename ...Args>
    explicit Var(Args &&...args) noexcept : Expr<T>{Variable::make_local_variable(type_desc<T>)} {
        std::vector<const Variable *> init{extract_variable(std::forward<Args>(args))...};
        Function::current().add_statement(std::make_unique<DeclareStmt>(this->_variable, std::move(init)));
    }
    
    using ExprBase::operator=;
    using ExprBase::operator+=;
    using ExprBase::operator-=;
    using ExprBase::operator*=;
    using ExprBase::operator/=;
    using ExprBase::operator%=;
    using ExprBase::operator<<=;
    using ExprBase::operator>>=;
    using ExprBase::operator|=;
    using ExprBase::operator&=;
    using ExprBase::operator^=;
};

// Deduction guides
template<typename T, std::enable_if_t<!is_expr<T>, int> = 0>
Var(T &&) -> Var<T>;

template<typename T>
Var(const Var<T> &) -> Var<T>;

template<typename T>
Var(Var<T> &&) -> Var<T>;

template<typename T>
Var(const Expr<T> &) -> Var<T>;

template<typename T>
Var(Expr<T> &&) -> Var<T>;

#define MAKE_UNARY_OP(op, op_tag)                                                                                       \
    template<typename T, std::enable_if_t<is_expr<T>, int> = 0>                                                         \
    auto operator op(T var) noexcept {                                                                                  \
        using R = std::decay_t<decltype(op std::declval<typename T::Type>())>;                                          \
        auto v = Variable::make_temporary(type_desc<R>, std::make_unique<UnaryExpr>(UnaryOp::op_tag, var.variable()));  \
        return Expr<R>{v};                                                                                              \
    }
MAKE_UNARY_OP(+, PLUS)
MAKE_UNARY_OP(-, MINUS)
MAKE_UNARY_OP(!, NOT)
MAKE_UNARY_OP(~, BIT_NOT)
#undef MAKE_UNARY_OP

#define MAKE_BINARY_OP(op, op_tag)                                                                                                                  \
    template<typename Lhs, typename Rhs, std::enable_if_t<std::disjunction_v<IsExpr<Lhs>, IsExpr<Rhs>>, int> = 0>                                   \
    inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {                                                                                        \
        Expr lhs_expr{lhs};                                                                                                                         \
        Expr rhs_expr{rhs};                                                                                                                         \
        using LhsT = typename decltype(lhs_expr)::Type;                                                                                             \
        using RhsT = typename decltype(rhs_expr)::Type;                                                                                             \
        using R = std::decay_t<decltype(std::declval<LhsT>() op std::declval<RhsT>())>;                                                             \
        auto v = Variable::make_temporary(type_desc<R>, std::make_unique<BinaryExpr>(BinaryOp::op_tag, lhs_expr.variable(), rhs_expr.variable()));  \
        return Var<R>{v};                                                                                                                           \
    }                                                                                                                                               \

MAKE_BINARY_OP(+, ADD)
MAKE_BINARY_OP(-, SUB)
MAKE_BINARY_OP(*, MUL)
MAKE_BINARY_OP(/, DIV)
MAKE_BINARY_OP(%, MOD)
MAKE_BINARY_OP(<<, SHL)
MAKE_BINARY_OP(>>, SHR)
MAKE_BINARY_OP(|, BIT_OR)
MAKE_BINARY_OP(&, BIT_AND)
MAKE_BINARY_OP(^, BIT_XOR)
MAKE_BINARY_OP(&&, AND)
MAKE_BINARY_OP(||, OR)
MAKE_BINARY_OP(==, EQUAL)
MAKE_BINARY_OP(!=, NOT_EQUAL)
MAKE_BINARY_OP(<, LESS)
MAKE_BINARY_OP(>, GREATER)
MAKE_BINARY_OP(<=, LESS_EQUAL)
MAKE_BINARY_OP(>=, GREATER_EQUAL)

#undef MAKE_BINARY_OP

inline auto thread_id() noexcept { return Expr<uint>{Variable::make_builtin(VariableTag::THREAD_ID)}; }
inline auto thread_xy() noexcept { return Expr<uint2>{Variable::make_builtin(VariableTag::THREAD_XY)}; }

template<typename T>
inline auto uniform(const T *p_data) noexcept { return Expr<T>{Variable::make_uniform_argument(type_desc<T>, p_data)}; }

template<typename T, std::enable_if_t<std::negation_v<std::is_pointer<T>>, int> = 0>
inline auto immutable(T data) noexcept {
    std::vector<std::byte> bytes(sizeof(T));
    std::memmove(bytes.data(), &data, bytes.size());
    return Expr<T>{Variable::make_uniform_argument(type_desc<T>, std::move(bytes))};
}

#define MAP_ARGUMENT_TO_TEMPLATE_ARGUMENT(arg) typename T##arg
#define MAP_ARGUMENT_TO_IS_EXPR(arg) IsExpr<T##arg>
#define MAP_ARGUMENT_TO_ARGUMENT(arg) T##arg &&arg
#define MAP_ARGUMENT_TO_EXPR(arg) Expr expr_##arg{arg};
#define MAP_ARGUMENT_TO_TRUE_TYPE(arg) using TT##arg = std::decay_t<typename decltype(expr_##arg)::Type>;
#define MAP_ARGUMENT_TO_TRUE_TYPE_DECLVAL(arg) std::declval<TT##arg &>()
#define MAP_ARGUMENT_TO_VARIABLE(arg) expr_##arg.variable()

#define MAKE_BUILTIN_FUNCTION_DEF(func, true_func, ...)                                                           \
template<LUISA_MAP_LIST(MAP_ARGUMENT_TO_TEMPLATE_ARGUMENT, __VA_ARGS__), std::enable_if_t<std::disjunction_v<     \
    LUISA_MAP_LIST(MAP_ARGUMENT_TO_IS_EXPR, __VA_ARGS__)>, int> = 0>                                              \
inline auto func(LUISA_MAP_LIST(MAP_ARGUMENT_TO_ARGUMENT, __VA_ARGS__)) noexcept {                                \
    LUISA_MAP(MAP_ARGUMENT_TO_EXPR, __VA_ARGS__)                                                                  \
    LUISA_MAP(MAP_ARGUMENT_TO_TRUE_TYPE, __VA_ARGS__)                                                             \
    using R = std::decay_t<decltype(true_func(LUISA_MAP_LIST(MAP_ARGUMENT_TO_TRUE_TYPE_DECLVAL, __VA_ARGS__)))>;  \
    return Expr<R>{Variable::make_temporary(type_desc<R>, std::make_unique<CallExpr>(                             \
        #func, std::vector<const Variable *>{LUISA_MAP_LIST(MAP_ARGUMENT_TO_VARIABLE, __VA_ARGS__)}))};           \
}

#define MAKE_BUILTIN_VOID_FUNCTION_DEF(func, ...)                                                                 \
template<LUISA_MAP_LIST(MAP_ARGUMENT_TO_TEMPLATE_ARGUMENT, __VA_ARGS__), std::enable_if_t<std::disjunction_v<     \
    LUISA_MAP_LIST(MAP_ARGUMENT_TO_IS_EXPR, __VA_ARGS__)>, int> = 0>                                              \
inline void func(LUISA_MAP_LIST(MAP_ARGUMENT_TO_ARGUMENT, __VA_ARGS__)) noexcept {                                \
    LUISA_MAP(MAP_ARGUMENT_TO_EXPR, __VA_ARGS__)                                                                  \
    Function::current().add_statement(std::make_unique<ExprStmt>(std::make_unique<CallExpr>(                      \
        #func, std::vector<const Variable *>{LUISA_MAP_LIST(MAP_ARGUMENT_TO_VARIABLE, __VA_ARGS__)})));           \
}

#define MAKE_BUILTIN_VOID_FUNCTION_VOID_DEF(func)                                                                 \
inline void func() noexcept {                                                                                     \
    Function::current().add_statement(std::make_unique<ExprStmt>(                                                 \
        std::make_unique<CallExpr>(#func, std::vector<const Variable *>{})));                                     \
}

MAKE_BUILTIN_FUNCTION_DEF(select, math::select, cond, tv, fv)
MAKE_BUILTIN_FUNCTION_DEF(sqrt, math::sqrt, x)
MAKE_BUILTIN_FUNCTION_DEF(sin, math::sin, x)
MAKE_BUILTIN_FUNCTION_DEF(cos, math::cos, x)
MAKE_BUILTIN_FUNCTION_DEF(tan, math::tan, x)
MAKE_BUILTIN_FUNCTION_DEF(asin, math::asin, x)
MAKE_BUILTIN_FUNCTION_DEF(acos, math::acos, x)
MAKE_BUILTIN_FUNCTION_DEF(atan, math::atan, x)
MAKE_BUILTIN_FUNCTION_DEF(atan2, math::atan2, y, x)
MAKE_BUILTIN_FUNCTION_DEF(ceil, math::ceil, x)
MAKE_BUILTIN_FUNCTION_DEF(floor, math::floor, x)
MAKE_BUILTIN_FUNCTION_DEF(round, math::round, x)
MAKE_BUILTIN_FUNCTION_DEF(pow, math::pow, x, y)
MAKE_BUILTIN_FUNCTION_DEF(exp, math::exp, x)
MAKE_BUILTIN_FUNCTION_DEF(log, math::log, x)
MAKE_BUILTIN_FUNCTION_DEF(log2, math::log2, x)
MAKE_BUILTIN_FUNCTION_DEF(log10, math::log10, x)
MAKE_BUILTIN_FUNCTION_DEF(min, math::min, x, y)
MAKE_BUILTIN_FUNCTION_DEF(max, math::max, x, y)
MAKE_BUILTIN_FUNCTION_DEF(abs, math::abs, x)
MAKE_BUILTIN_FUNCTION_DEF(clamp, math::clamp, x, a, b)
MAKE_BUILTIN_FUNCTION_DEF(lerp, math::lerp, a, b, t)
MAKE_BUILTIN_FUNCTION_DEF(radians, math::radians, deg)
MAKE_BUILTIN_FUNCTION_DEF(degrees, math::degrees, rad)
MAKE_BUILTIN_FUNCTION_DEF(normalize, math::normalize, v)
MAKE_BUILTIN_FUNCTION_DEF(length, math::length, v)
MAKE_BUILTIN_FUNCTION_DEF(dot, math::dot, u, v)
MAKE_BUILTIN_FUNCTION_DEF(cross, math::cross, u, v)

MAKE_BUILTIN_FUNCTION_DEF(any, vector::any, v)
MAKE_BUILTIN_FUNCTION_DEF(all, vector::all, v)
MAKE_BUILTIN_FUNCTION_DEF(none, vector::none, v)

MAKE_BUILTIN_FUNCTION_DEF(make_float3x3, matrix::make_float3x3, val_or_mat4)
MAKE_BUILTIN_FUNCTION_DEF(make_float3x3, matrix::make_float3x3, c0, c1, c2)
MAKE_BUILTIN_FUNCTION_DEF(make_float3x3, matrix::make_float3x3, m00, m01, m02, m10, m11, m12, m20, m21, m22)

MAKE_BUILTIN_FUNCTION_DEF(make_float4x4, matrix::make_float4x4, val_or_mat3)
MAKE_BUILTIN_FUNCTION_DEF(make_float4x4, matrix::make_float4x4, c0, c1, c2, c3)
MAKE_BUILTIN_FUNCTION_DEF(make_float4x4, matrix::make_float4x4, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33)

MAKE_BUILTIN_FUNCTION_DEF(inverse, math::inverse, m)
MAKE_BUILTIN_FUNCTION_DEF(transpose, math::transpose, m)

// make_vec2
#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC2(T)                                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##2, vector::make_##T##2, v)                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##2, vector::make_##T##2, x, y)             \

// make_vec3
#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC3(T)                                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##3, vector::make_##T##3, v)                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##3, vector::make_##T##3, x, y)             \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##3, vector::make_##T##3, x, y, z)          \

// make_vec4
#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC4(T)                                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##4, vector::make_##T##4, v)                \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##4, vector::make_##T##4, x, y)             \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##4, vector::make_##T##4, x, y, z)          \
MAKE_BUILTIN_FUNCTION_DEF(make_##T##4, vector::make_##T##4, x, y, z, w)       \

// make_packed_vec3
#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_PACKED_VEC3(T)                         \
MAKE_BUILTIN_FUNCTION_DEF(make_packed_##T##3, vector::make_packed_##T##3, v)  \

#define MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(T)                                 \
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC2(T)                                        \
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC3(T)                                        \
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC4(T)                                        \
MAKE_BUILTIN_FUNCTION_DEF_MAKE_PACKED_VEC3(T)                                 \

MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(bool)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(float)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(char)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(uchar)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(short)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(ushort)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(int)
MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC(uint)

#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC
#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC4
#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC3
#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_VEC2
#undef MAKE_BUILTIN_FUNCTION_DEF_MAKE_PACKED_VEC3

MAKE_BUILTIN_VOID_FUNCTION_DEF(atomic_store, object, desired)

MAKE_BUILTIN_FUNCTION_DEF(atomic_load, atomic::atomic_load, object)
MAKE_BUILTIN_FUNCTION_DEF(atomic_exchange, atomic::atomic_exchange, object, desired)
MAKE_BUILTIN_FUNCTION_DEF(atomic_compare_exchange_weak, atomic::atomic_compare_exchange_weak, object, expected, desired)
MAKE_BUILTIN_FUNCTION_DEF(atomic_compare_exchange_strong, atomic::atomic_compare_exchange_strong, object, expected, desired)
MAKE_BUILTIN_FUNCTION_DEF(atomic_fetch_add, atomic::atomic_fetch_add, object, operand)
MAKE_BUILTIN_FUNCTION_DEF(atomic_fetch_sub, atomic::atomic_fetch_sub, object, operand)
MAKE_BUILTIN_FUNCTION_DEF(atomic_fetch_and, atomic::atomic_fetch_and, object, operand)
MAKE_BUILTIN_FUNCTION_DEF(atomic_fetch_or, atomic::atomic_fetch_or, object, operand)
MAKE_BUILTIN_FUNCTION_DEF(atomic_fetch_xor, atomic::atomic_fetch_xor, object, operand)

MAKE_BUILTIN_VOID_FUNCTION_VOID_DEF(threadgroup_barrier)

#undef MAP_ARGUMENT_TO_TEMPLATE_ARGUMENT
#undef MAP_ARGUMENT_TO_IS_EXPR
#undef MAP_ARGUMENT_TO_ARGUMENT
#undef MAP_ARGUMENT_TO_EXPR
#undef MAP_ARGUMENT_TO_TRUE_TYPE
#undef MAP_ARGUMENT_TO_TRUE_TYPE_DECLVAL
#undef MAP_ARGUMENT_TO_VARIABLE
#undef MAKE_BUILTIN_FUNCTION_DEF
#undef MAKE_BUILTIN_VOID_FUNCTION_DEF
#undef MAKE_BUILTIN_VOID_FUNCTION_VOID_DEF

template<typename T>
[[nodiscard]] inline auto cast(const ExprBase &v) {
    return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<CastExpr>(CastOp::STATIC, v.variable(), type_desc<T>))};
}

template<typename T>
[[nodiscard]] inline auto reinterpret(const ExprBase &v) {
    return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<CastExpr>(CastOp::REINTERPRET, v.variable(), type_desc<T>))};
}

template<typename T>
[[nodiscard]] inline auto bitcast(const ExprBase &v) {
    return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<CastExpr>(CastOp::BITWISE, v.variable(), type_desc<T>))};
}

}

#define LUISA_STRUCT_BEGIN(S)                                                                    \
namespace luisa::compute::dsl {                                                                  \
    template<>                                                                                   \
    struct Structure<S> {                                                                        \
        [[nodiscard]] static TypeDesc *desc() noexcept {                                         \
            using This = S;                                                                      \
            static TypeDesc td;                                                                  \
            static std::once_flag flag;                                                          \
            std::call_once(flag, []{                                                             \
                td.type = TypeCatalog::STRUCTURE;                                                \
                td.member_names.clear();                                                         \
                td.member_types.clear();                                                         \

#define LUISA_STRUCT_MEMBER(member)                                                              \
                td.member_names.emplace_back(#member);                                           \
                td.member_types.emplace_back(type_desc<decltype(std::declval<This>().member)>);  \

#define LUISA_STRUCT_END()                                                                       \
            });                                                                                  \
            return &td;                                                                          \
        }                                                                                        \
    };                                                                                           \
}                                                                                                \

#define LUISA_STRUCT_MAP_MEMBER_NAME_TO_EXPR(name)                                                            \
    [[nodiscard]] auto name() const noexcept {                                                                \
        using R = std::decay_t<decltype(std::declval<Type>().name)>;                                          \
        return Variable::make_temporary(type_desc<R>, std::make_unique<MemberExpr>(this->_variable, #name));  \
    }

#define LUISA_STRUCT_SPECIALIZE_EXPR(S, ...)                            \
namespace luisa::compute::dsl {                                         \
template<> struct Expr<S> : public ExprBase {                           \
    using Type = S;                                                     \
    explicit Expr(const Variable *v) noexcept : ExprBase{v} {}          \
    Expr(ExprBase &&expr) noexcept : ExprBase{expr.variable()} {}       \
    Expr(const ExprBase &expr) noexcept : ExprBase{expr.variable()} {}  \
    LUISA_MAP(LUISA_STRUCT_MAP_MEMBER_NAME_TO_EXPR, __VA_ARGS__)        \
};                                                                      \
}

#define LUISA_STRUCT(S, ...)                            \
LUISA_STRUCT_BEGIN(S)                                   \
     LUISA_MAP(LUISA_STRUCT_MEMBER, __VA_ARGS__)        \
LUISA_STRUCT_END()                                      \
LUISA_STRUCT_SPECIALIZE_EXPR(S, __VA_ARGS__)

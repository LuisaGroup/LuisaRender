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
    using Type = T;
    explicit Expr(const Variable *v) noexcept: ExprBase{v} {}
    Expr(T v) noexcept: ExprBase{Variable::make_temporary(type_desc<T>, std::make_unique<ValueExpr>(v))} {}
    Expr(ExprBase &&expr) noexcept: ExprBase{expr.variable()} {}
    Expr(const ExprBase &expr) noexcept: ExprBase{expr.variable()} {}
};

template<typename T, size_t N>
struct Expr<std::array<T, N>> : public ExprBase {
    using Type = std::array<T, N>;
    explicit Expr(const Variable *v) noexcept: ExprBase{v} {}
    Expr(ExprBase &&expr) noexcept: ExprBase{expr.variable()} {}
    Expr(const ExprBase &expr) noexcept: ExprBase{expr.variable()} {}
    
    [[nodiscard]] auto operator[](const ExprBase &index) const noexcept {
        return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, _variable, index.variable()))};
    }
};

template<typename T, size_t N>
struct Expr<T[N]> : public ExprBase {
    using Type = std::array<T, N>;
    explicit Expr(const Variable *v) noexcept: ExprBase{v} {}
    Expr(ExprBase &&expr) noexcept: ExprBase{expr.variable()} {}
    Expr(const ExprBase &expr) noexcept: ExprBase{expr.variable()} {}
    [[nodiscard]] auto operator[](const ExprBase &index) const noexcept {
        return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, _variable, index.variable()))};
    }
};

template<typename T, bool packed>
struct Expr<Vector<T, 2, packed>> : public ExprBase {
    using Type = Vector<T, 2, packed>;
    explicit Expr(const Variable *v) noexcept: ExprBase{v} {}
    Expr(ExprBase &&expr) noexcept: ExprBase{expr.variable()} {}
    Expr(const ExprBase &expr) noexcept: ExprBase{expr.variable()} {}
    [[nodiscard]] auto operator[](const ExprBase &index) const noexcept {
        return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, _variable, index.variable()))};
    }
    [[nodiscard]] auto x() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "x"))}; }
    [[nodiscard]] auto y() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "y"))}; }
    [[nodiscard]] auto r() const noexcept { return x(); }
    [[nodiscard]] auto g() const noexcept { return y(); }
};

template<typename T, bool packed>
struct Expr<Vector<T, 3, packed>> : public ExprBase {
    using Type = Vector<T, 3, packed>;
    explicit Expr(const Variable *v) noexcept: ExprBase{v} {}
    Expr(ExprBase &&expr) noexcept: ExprBase{expr.variable()} {}
    Expr(const ExprBase &expr) noexcept: ExprBase{expr.variable()} {}
    [[nodiscard]] auto operator[](const ExprBase &index) const noexcept {
        return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, _variable, index.variable()))};
    }
    [[nodiscard]] auto x() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "x"))}; }
    [[nodiscard]] auto y() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "y"))}; }
    [[nodiscard]] auto z() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "z"))}; }
    [[nodiscard]] auto r() const noexcept { return x(); }
    [[nodiscard]] auto g() const noexcept { return y(); }
    [[nodiscard]] auto b() const noexcept { return z(); }
};

template<typename T, bool packed>
struct Expr<Vector<T, 4, packed>> : public ExprBase {
    using Type = Vector<T, 4, packed>;
    explicit Expr(const Variable *v) noexcept: ExprBase{v} {}
    Expr(ExprBase &&expr) noexcept: ExprBase{expr.variable()} {}
    Expr(const ExprBase &expr) noexcept: ExprBase{expr.variable()} {}
    [[nodiscard]] auto operator[](const ExprBase &index) const noexcept {
        return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, _variable, index.variable()))};
    }
    [[nodiscard]] auto x() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "x"))}; }
    [[nodiscard]] auto y() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "y"))}; }
    [[nodiscard]] auto z() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "z"))}; }
    [[nodiscard]] auto w() const noexcept { return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<MemberExpr>(_variable, "w"))}; }
    [[nodiscard]] auto r() const noexcept { return x(); }
    [[nodiscard]] auto g() const noexcept { return y(); }
    [[nodiscard]] auto b() const noexcept { return z(); }
    [[nodiscard]] auto a() const noexcept { return w(); }
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
    Var(Args &&...args) noexcept : Expr<T>{Variable::make_local_variable(type_desc<T>)} {
        std::vector<const Variable *> init{detail::extract_variable(std::forward<Args>(args))...};
        Function::current().add_statement(std::make_unique<DeclareStmt>(Expr<T>::_variable, std::move(init)));
    }

#define MAKE_ASSIGN_OP(op)  \
    template<typename U>    \
    void operator op(U &&rhs) noexcept { ExprBase::operator op(Expr{std::forward<U>(rhs)}); }
    
    MAKE_ASSIGN_OP(=);
    MAKE_ASSIGN_OP(+=);
    MAKE_ASSIGN_OP(-=);
    MAKE_ASSIGN_OP(*=);
    MAKE_ASSIGN_OP(/=);
    MAKE_ASSIGN_OP(%=);
    MAKE_ASSIGN_OP(<<=);
    MAKE_ASSIGN_OP(>>=);
    MAKE_ASSIGN_OP(|=);
    MAKE_ASSIGN_OP(&=);
    MAKE_ASSIGN_OP(^=);
#undef MAKE_ASSIGN_OP

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

template<typename T>
class Threadgroup {

private:
    const Variable *_variable;

public:
    explicit Threadgroup(uint n) noexcept: _variable{Variable::make_threadgroup_variable(type_desc<T>, n)} {}
    
    template<typename Index>
    [[nodiscard]] auto operator[](Index &&index) const noexcept {
        Expr i{std::forward<Index>(index)};
        return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, _variable, i.variable()))};
    }
};

#define MAKE_UNARY_OP(op, op_tag)                                                                                            \
    template<typename T, std::enable_if_t<is_expr<T>, int> = 0>                                                              \
    auto operator op(T &&var) noexcept {                                                                                     \
        Expr var_expr{std::forward<T>(var)};                                                                                 \
        using R = std::decay_t<decltype(op std::declval<typename std::decay_t<decltype(var_expr)>::Type>())>;                \
        auto v = Variable::make_temporary(type_desc<R>, std::make_unique<UnaryExpr>(UnaryOp::op_tag, var_expr.variable()));  \
        return Expr<R>{v};                                                                                                   \
    }
MAKE_UNARY_OP(+, PLUS)
MAKE_UNARY_OP(-, MINUS)
MAKE_UNARY_OP(!, NOT)
MAKE_UNARY_OP(~, BIT_NOT)
#undef MAKE_UNARY_OP

#define MAKE_BINARY_OP(op, op_tag)                                                                                                                  \
    template<typename Lhs, typename Rhs, std::enable_if_t<std::disjunction_v<IsExpr<Lhs>, IsExpr<Rhs>>, int> = 0>                                   \
    inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {                                                                                        \
        Expr lhs_expr{std::forward<Lhs>(lhs)};                                                                                                      \
        Expr rhs_expr{std::forward<Rhs>(rhs)};                                                                                                      \
        using LhsT = typename decltype(lhs_expr)::Type;                                                                                             \
        using RhsT = typename decltype(rhs_expr)::Type;                                                                                             \
        using R = std::decay_t<decltype(std::declval<LhsT>() op std::declval<RhsT>())>;                                                             \
        auto v = Variable::make_temporary(type_desc<R>, std::make_unique<BinaryExpr>(BinaryOp::op_tag, lhs_expr.variable(), rhs_expr.variable()));  \
        return Expr<R>{v};                                                                                                                          \
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
    return Expr<T>{Variable::make_immutable_argument(type_desc<T>, std::move(bytes))};
}

#define MAP_ARGUMENT_TO_TEMPLATE_ARGUMENT(arg) typename T##arg
#define MAP_ARGUMENT_TO_ARGUMENT(arg) T##arg &&arg
#define MAP_ARGUMENT_TO_EXPR(arg) Expr expr_##arg{arg};
#define MAP_ARGUMENT_TO_TRUE_TYPE(arg) using TT##arg = std::decay_t<typename decltype(expr_##arg)::Type>;
#define MAP_ARGUMENT_TO_TRUE_TYPE_DECLVAL(arg) std::declval<TT##arg &>()
#define MAP_ARGUMENT_TO_VARIABLE(arg) expr_##arg.variable()

#define MAKE_BUILTIN_FUNCTION_DEF(func, true_func, ...)                                                           \
template<LUISA_MAP_LIST(MAP_ARGUMENT_TO_TEMPLATE_ARGUMENT, __VA_ARGS__)>                                          \
inline auto func(LUISA_MAP_LIST(MAP_ARGUMENT_TO_ARGUMENT, __VA_ARGS__)) noexcept {                                \
    LUISA_MAP(MAP_ARGUMENT_TO_EXPR, __VA_ARGS__)                                                                  \
    LUISA_MAP(MAP_ARGUMENT_TO_TRUE_TYPE, __VA_ARGS__)                                                             \
    using R = std::decay_t<decltype(true_func(LUISA_MAP_LIST(MAP_ARGUMENT_TO_TRUE_TYPE_DECLVAL, __VA_ARGS__)))>;  \
    return Expr<R>{Variable::make_temporary(type_desc<R>, std::make_unique<CallExpr>(                             \
        #func, std::vector<const Variable *>{LUISA_MAP_LIST(MAP_ARGUMENT_TO_VARIABLE, __VA_ARGS__)}))};           \
}

#define MAKE_BUILTIN_VOID_FUNCTION_DEF(func, ...)                                                                 \
template<LUISA_MAP_LIST(MAP_ARGUMENT_TO_TEMPLATE_ARGUMENT, __VA_ARGS__)>                                          \
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

#define LUISA_STRUCT_MAP_MEMBER_NAME_TO_EXPR(name)                                                                     \
    [[nodiscard]] auto name() const noexcept {                                                                         \
        using R = std::decay_t<decltype(std::declval<Type>().name)>;                                                   \
        return Expr<R>{Variable::make_temporary(type_desc<R>, std::make_unique<MemberExpr>(this->_variable, #name))};  \
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

// Let's define some syntax sugars...
#ifndef LUISA_DISABLE_DSL_SYNTAX_SUGARS

namespace luisa::compute::dsl {

class IfStmtBuilder {

private:
    const Variable *_cond{nullptr};
    std::function<void()> _true;
    std::function<void()> _false;

public:
    template<typename Cond>
    explicit IfStmtBuilder(Cond &&cond) noexcept {
        Expr cond_expr{std::forward<Cond>(cond)};
        _cond = cond_expr.variable();
    }
    
    ~IfStmtBuilder() noexcept {
        if (_false) {
            Function::current().add_statement(std::make_unique<IfStmt>(_cond, std::move(_true), std::move(_false)));
        } else {
            Function::current().add_statement(std::make_unique<IfStmt>(_cond, std::move(_true)));
        }
    }
    
    IfStmtBuilder &operator<<(std::function<void()> t) noexcept {
        _true = std::move(t);
        return *this;
    }
    
    void operator>>(std::function<void()> f) noexcept { _false = std::move(f); }
    
    [[nodiscard]] IfStmtBuilder &operator<<(IfStmtBuilder *elif) noexcept {
        _false = [elif] { delete elif; };
        return *elif;
    }
};

#define If(...) ::luisa::compute::dsl::IfStmtBuilder{__VA_ARGS__} << [&]()
#define Else >> [&]()
#define Elif(...) << (new ::luisa::compute::dsl::IfStmtBuilder{__VA_ARGS__}) << [&]()

class WhileStmtBuilder {

private:
    const Variable *_cond{nullptr};
    std::function<void()> _body;

public:
    template<typename Cond>
    explicit WhileStmtBuilder(Cond &&cond) noexcept {
        Expr cond_expr{std::forward<Cond>(cond)};
        _cond = cond_expr.variable();
    }
    
    ~WhileStmtBuilder() noexcept { Function::current().add_statement(std::make_unique<WhileStmt>(_cond, std::move(_body))); }
    
    void operator<<(std::function<void()> body) noexcept { _body = std::move(body); }
};

#define While(...) ::luisa::compute::dsl::WhileStmtBuilder{__VA_ARGS__} << [&]()

class DoWhileStmtBuilder {

private:
    const Variable *_cond{nullptr};
    std::function<void()> _body;

public:
    ~DoWhileStmtBuilder() noexcept { Function::current().add_statement(std::make_unique<DoWhileStmt>(std::move(_body), _cond)); }
    
    DoWhileStmtBuilder &operator<<(std::function<void()> body) noexcept {
        _body = std::move(body);
        return *this;
    }
    
    void operator<<(const Variable *cond) noexcept { _cond = cond; }
};

#define Do ::luisa::compute::dsl::DoWhileStmtBuilder{} << [&]()
#define When(...) << ::luisa::compute::dsl::detail::extract_variable(__VA_ARGS__)

class SwitchStmtBuilder {

private:
    const Variable *_expr{nullptr};
    
public:
    template<typename T>
    explicit SwitchStmtBuilder(T &&expr) noexcept {
        Expr e{expr};
        _expr = e.variable();
    }
    
    template<typename F>
    void operator<<(F &&f) noexcept {
        Function::current().add_statement(std::make_unique<SwitchStmt>(_expr, std::forward<F>(f)));
    }
};

class SwitchCaseStmtBuilder {

private:
    const Variable *_expr{nullptr};

public:
    template<typename T>
    explicit SwitchCaseStmtBuilder(T &&expr) noexcept {
        Expr e{expr};
        _expr = e.variable();
    }
    
    template<typename F>
    void operator<<(F &&f) noexcept {
        Function::current().add_statement(std::make_unique<SwitchCaseStmt>(_expr, std::forward<F>(f)));
    }
};

struct SwitchDefaultStmtBuilder {
    template<typename F>
    void operator<<(F &&f) noexcept {
        Function::current().add_statement(std::make_unique<SwitchDefaultStmt>(std::forward<F>(f)));
    }
};

#define Switch(...) ::luisa::compute::dsl::SwitchStmtBuilder{__VA_ARGS__} << [&]()
#define Case(...) ::luisa::compute::dsl::SwitchCaseStmtBuilder{__VA_ARGS__} << [&]()
#define Default ::luisa::compute::dsl::SwitchDefaultStmtBuilder{} << [&]()

#define Return ::luisa::compute::dsl::Function::current().add_return()
#define Break ::luisa::compute::dsl::Function::current().add_break()
#define Continue ::luisa::compute::dsl::Function::current().add_continue()

}

#endif

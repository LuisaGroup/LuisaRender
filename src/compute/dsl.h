//
// Created by Mike Smith on 2020/8/5.
//

#pragma once

#include <type_traits>

#include <core/math_util.h>
#include <compute/variable.h>
#include <compute/function.h>
#include <compute/expression.h>
#include <compute/statement.h>

namespace luisa::compute::dsl {

template<typename T>
struct Var;

namespace detail {

template<typename T>
auto is_var_impl(T *) noexcept { return std::false_type{}; }

template<typename T>
auto is_var_impl(const Var<T> *) noexcept { return std::true_type{}; }

}

template<typename T>
constexpr auto is_var = decltype(detail::is_var_impl(std::declval<T *>()))::value;

template<typename T>
class Var {

private:
    const Variable *_variable{nullptr};

public:
    explicit Var(const Variable *variable) noexcept: _variable{variable} {}
    
    template<typename ...Args>
    explicit Var(Args &&...args) {
        
        constexpr auto extract_variable = [](auto &&v) noexcept -> const Variable * {
            using Type = std::decay_t<decltype(v)>;
            if constexpr (is_var<Type>) {
                return v.variable();
            } else {
                return Variable::make_temporary(type_desc<Type>, std::make_unique<ValueExpr>(v));
            }
        };
        
        std::vector<const Variable *> initializers{extract_variable(args)...};
        _variable = Variable::make_local_variable(type_desc<T>);
        Function::current().add_statement(std::make_unique<DeclareStmt>(_variable, std::move(initializers)));
    }
    
    Var(Var<T> &&another) noexcept: _variable{another._variable} {}
    
    [[nodiscard]] const Variable *variable() const noexcept { return _variable; }

#define MAKE_ASSIGN_OP(op, op_tag)                                                                                     \
    template<typename U, typename = decltype(std::declval<T &>() op std::declval<U>())>                                \
    void operator op(const Var<U> &rhs) const noexcept {                                                               \
        Function::current().add_statement(std::make_unique<AssignStmt>(AssignOp::op_tag, _variable, rhs.variable()));  \
        _variable->mark_write();                                                                                       \
        rhs->variable()->mark_read();                                                                                  \
    }                                                                                                                  \

    MAKE_ASSIGN_OP(=, ASSIGN)
    MAKE_ASSIGN_OP(+=, ADD_ASSIGN)
    MAKE_ASSIGN_OP(-=, SUB_ASSIGN)
    MAKE_ASSIGN_OP(*=, MUL_ASSIGN)
    MAKE_ASSIGN_OP(/=, DIV_ASSIGN)
    MAKE_ASSIGN_OP(%=, MOD_ASSIGN)
    MAKE_ASSIGN_OP(^=, BIT_XOR_ASSIGN)
    MAKE_ASSIGN_OP(|=, BIT_OR_ASSIGN)
    MAKE_ASSIGN_OP(&=, BIT_AND_ASSIGN)
    MAKE_ASSIGN_OP(<<=, SHL_ASSIGN)
    MAKE_ASSIGN_OP(>>=, SHR_ASSIGN)

#undef MAKE_ASSIGN_OP

#define MAKE_BINARY_OP(op, op_tag, op_end)                                                                                           \
    template<typename U, typename = decltype(std::declval<T>() op std::declval<U>() op_end)>                                         \
    auto operator op op_end(const Var<U> &rhs) const noexcept {                                                                      \
        using R = std::decay_t<decltype(std::declval<T>() op std::declval<U>() op_end)>;                                             \
        auto v = Variable::make_temporary(type_desc<R>, std::make_unique<BinaryExpr>(BinaryOp::op_tag, _variable, rhs.variable()));  \
        _variable->mark_read();                                                                                                      \
        rhs->variable()->mark_read();                                                                                                \
        return Var<R>{v};                                                                                                            \
    }                                                                                                                                \
    template<typename U, typename = decltype(std::declval<T>() op std::declval<U>() op_end)>                                         \
    auto operator op op_end(U &&rhs) const noexcept {                                                                                \
        using R = std::decay_t<decltype(std::declval<T>() op rhs op_end)>;                                                           \
        auto rhs_v = Variable::make_temporary(type_desc<std::decay_t<U>>, std::make_unique<ValueExpr>(std::forward<U>(rhs)));        \
        auto v = Variable::make_temporary(type_desc<R>, std::make_unique<BinaryExpr>(BinaryOp::op_tag, _variable, rhs_v));           \
        _variable->mark_read();                                                                                                      \
        rhs->variable()->mark_read();                                                                                                \
        return Var<R>{v};                                                                                                            \
    }                                                                                                                                \

    MAKE_BINARY_OP(+, ADD,)
    MAKE_BINARY_OP(-, SUB,)
    MAKE_BINARY_OP(*, MUL,)
    MAKE_BINARY_OP(/, DIV,)
    MAKE_BINARY_OP(%, MOD,)
    MAKE_BINARY_OP(<<, SHL,)
    MAKE_BINARY_OP(>>, SHR,)
    MAKE_BINARY_OP(|, BIT_OR,)
    MAKE_BINARY_OP(&, BIT_AND,)
    MAKE_BINARY_OP(^, BIT_XOR,)
    MAKE_BINARY_OP(&&, AND,)
    MAKE_BINARY_OP(||, OR,)
    MAKE_BINARY_OP(==, EQUAL,)
    MAKE_BINARY_OP(!=, NOT_EQUAL,)
    MAKE_BINARY_OP(<, LESS,)
    MAKE_BINARY_OP(>, GREATER,)
    MAKE_BINARY_OP(<=, LESS_EQUAL,)
    MAKE_BINARY_OP(>=, GREATER_EQUAL,)
    MAKE_BINARY_OP([, ACCESS, ])

#undef MAKE_BINARY_OP

    [[nodiscard]] auto x() const noexcept {
        if constexpr (is_vector<T>) {
            using MemberType = std::decay_t<decltype(std::declval<T>().x)>;
            auto v = Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, "x"));
            return Var<MemberType>{v};
        } else { LUISA_ERROR("This variable has no component x."); }
    }
    
    [[nodiscard]] auto y() const noexcept {
        if constexpr (is_vector<T>) {
            using MemberType = std::decay_t<decltype(std::declval<T>().y)>;
            auto v = Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, "y"));
            return Var<MemberType>{v};
        } else { LUISA_ERROR("This variable has no component y."); }
    }
    
    [[nodiscard]] auto z() const noexcept {
        if constexpr (is_vector_3<T> || is_vector_4<T>) {
            using MemberType = std::decay_t<decltype(std::declval<T>().z)>;
            auto v = Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, "z"));
            return Var<MemberType>{v};
        } else { LUISA_ERROR("This variable has no component z."); }
    }
    [[nodiscard]] auto w() const noexcept {
        if constexpr (is_vector_4<T>) {
            using MemberType = std::decay_t<decltype(std::declval<T>().w)>;
            auto v = Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, "w"));
            return Var<MemberType>{v};
        } else { LUISA_ERROR("This variable has no component w."); }
    }
    
    [[nodiscard]] auto r() const noexcept { return x(); }
    [[nodiscard]] auto g() const noexcept { return y(); }
    [[nodiscard]] auto b() const noexcept { return z(); }
    [[nodiscard]] auto a() const noexcept { return w(); }
};

// Deduction guides
template<typename T>
Var(const Var<T> &) -> Var<T>;

template<typename T>
Var(Var<T> &&) -> Var<T>;

template<typename T, std::enable_if_t<!is_var<T>, int> = 0>
Var(T &&) -> Var<T>;

#define MAKE_UNARY_OP(op, op_tag)                                                                                       \
    template<typename T, typename = decltype(op std::declval<T>())>                                                     \
    auto operator op(const Var<T> &var) noexcept {                                                                      \
        using R = std::decay_t<decltype(!std::declval<T>())>;                                                           \
        auto v = Variable::make_temporary(type_desc<R>, std::make_unique<UnaryExpr>(UnaryOp::op_tag, var.variable()));  \
        var.variable()->mark_read();                                                                                    \
        return Var<R>{v};                                                                                               \
    }                                                                                                                   \

MAKE_UNARY_OP(+, PLUS)
MAKE_UNARY_OP(-, MINUS)
MAKE_UNARY_OP(!, NOT)
MAKE_UNARY_OP(~, BIT_NOT)

#undef MAKE_UNARY_OP

#define MAKE_BINARY_OP(op, op_tag)                                                                                                 \
    template<typename Lhs, typename Rhs, typename = decltype(std::declval<Lhs>() op std::declval<Rhs>())>                          \
    inline auto operator op(Lhs &&lhs, const Var<Rhs> &rhs) noexcept {                                                             \
        using R = std::decay_t<decltype(lhs op std::declval<Rhs>())>;                                                              \
        auto lhs_v = Variable::make_temporary(type_desc<std::decay_t<Lhs>>, std::make_unique<ValueExpr>(std::forward<Lhs>(lhs)));  \
        auto v = Variable::make_temporary(type_desc<R>, std::make_unique<BinaryExpr>(BinaryOp::op_tag, lhs_v, rhs.variable()));    \
        lhs_v->mark_read();                                                                                                        \
        rhs.variable()->mark_read();                                                                                               \
        return Var<R>{v};                                                                                                          \
    }                                                                                                                              \

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

inline Var<uint> thread_id() noexcept { return Var<uint>{Variable::make_builtin(VariableTag::THREAD_ID)}; }
inline Var<uint2> thread_xy() noexcept { return Var<uint2>{Variable::make_builtin(VariableTag::THREAD_XY)}; }

template<typename T>
inline Var<T> uniform(const T *p_data) noexcept { return Var<T>{Variable::make_uniform_argument(type_desc<T>, p_data)}; }

template<typename T, std::enable_if_t<std::negation_v<std::is_pointer<T>>, int> = 0>
inline Var<T> immutable(T data) noexcept {
    std::vector<std::byte> bytes(sizeof(T));
    std::memmove(bytes.data(), &data, bytes.size());
    return Var<T>{Variable::make_uniform_argument(type_desc<T>, std::move(bytes))};
}

template<typename T>
struct Expr : public Var<T> {
    using Type = T;
    Expr(const Var<T> &var) noexcept : Var<T>{var.variable()} {}
    Expr(T v) noexcept : Var<T>{Variable::make_temporary(type_desc<T>, std::make_unique<ValueExpr>(v))} {}
};

// Deduction guides
template<typename T>
Expr(const Var<T> &) -> Expr<T>;

template<typename T, std::enable_if_t<!is_var<T>, int> = 0>
Expr(T &&) -> Expr<T>;


template<typename Tx, typename Ty>
inline auto min(Tx x, Ty y) noexcept {
    Expr expr_x{x};
    Expr expr_y{y};
    using TTx = std::decay_t<typename decltype(expr_x)::Type>;
    using TTy = std::decay_t<typename decltype(expr_x)::Type>;
    using R = std::decay_t<decltype(math::min(std::declval<TTx>(), std::declval<TTy>()))>;
    auto v = Variable::make_temporary(type_desc<R>, std::make_unique<CallExpr("min", {expr_x.variable(), expr_y.variable()})>);
    return Var<R>{v};
};

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

#define LUISA_STRUCT_MEMBER_TO_VAR(name)                                                                       \
[[nodiscard]] auto name() const noexcept {                                                                     \
    using MemberType = std::decay_t<decltype(std::declval<This>().name)>;                                      \
    auto v = Variable::make_temporary(type_desc<MemberType>, std::make_unique<MemberExpr>(_variable, #name));  \
    return Var<MemberType>{v};                                                                                 \
}                                                                                                              \

#define LUISA_STRUCT_VAR_SPECIALIZE(S, ...)                                                                    \
namespace luisa::compute::dsl {                                                                                \
template<> class Var<S> {                                                                                      \
private:                                                                                                       \
    using This = S;                                                                                            \
    const Variable *_variable{nullptr};                                                                        \
public:                                                                                                        \
    explicit Var(const Variable *variable) noexcept: _variable{variable} {}                                    \
    template<typename ...Args> explicit Var(Args &&...args) {                                                  \
        constexpr auto extract_variable = [](auto &&v) noexcept -> const Variable * {                          \
            using Type = std::decay_t<decltype(v)>;                                                            \
            if constexpr (is_var<Type>) { return v.variable(); }                                               \
            else { return Variable::make_temporary(type_desc<Type>, std::make_unique<ValueExpr>(v)); }         \
        };                                                                                                     \
        std::vector<const Variable *> initializers{extract_variable(args)...};                                 \
        _variable = Variable::make_local_variable(type_desc<S>);                                               \
        Function::current().add_statement(std::make_unique<DeclareStmt>(_variable, std::move(initializers)));  \
    }                                                                                                          \
    Var(Var<S> &&another) noexcept: _variable{another._variable} {}                                            \
    [[nodiscard]] const Variable *variable() const noexcept { return _variable; }                              \
    LUISA_MAP(LUISA_STRUCT_MEMBER_TO_VAR, __VA_ARGS__)                                                         \
};                                                                                                             \
}                                                                                                              \

#define LUISA_STRUCT(S, ...)                            \
LUISA_STRUCT_BEGIN(S)                                   \
     LUISA_MAP(LUISA_STRUCT_MEMBER, __VA_ARGS__)        \
LUISA_STRUCT_END()                                      \
LUISA_STRUCT_VAR_SPECIALIZE(S, __VA_ARGS__)             \

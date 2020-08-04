//
// Created by Mike Smith on 2020/7/30.
//

#pragma once

#include <compute/function.h>
#include <compute/expr_helpers.h>

namespace luisa::dsl {

inline void Void(Variable v) { Function::current().add_statement(std::make_unique<ExprStmt>(v)); }

#define MAKE_VARIABLE_ASSIGNMENT_OPERATOR_IMPL(op)                            \
template<typename T, detail::EnableIfLiteralOperand<T>>                       \
inline void Variable::operator op(T &&rhs) const noexcept{                    \
    this->operator op(literal(std::forward<T>(rhs)));                         \
}                                                                             \

LUISA_MAP_MACRO(MAKE_VARIABLE_ASSIGNMENT_OPERATOR_IMPL, =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=)
#undef MAKE_VARIABLE_ASSIGNMENT_OPERATOR_IMPL

template<typename T>
[[nodiscard]] inline Variable arg() noexcept {
    return Function::current().arg<T>();
}

template<typename T, typename ...Literals>
[[nodiscard]] inline Variable var(Literals &&...vs) noexcept {
    return Function::current().var<T>(std::forward<Literals>(vs)...);
}

[[nodiscard]] inline Variable var(LiteralExpr::Value v) noexcept {
    return Function::current().var(std::move(v));
}

template<typename T, typename ...Literals>
[[nodiscard]] inline Variable let(Literals &&...vs) noexcept {
    return Function::current().constant<T>(std::forward<Literals>(vs)...);
}

[[nodiscard]] inline Variable let(LiteralExpr::Value v) noexcept {
    return Function::current().constant(std::move(v));
}

#define MAKE_VARIABLE_ASSIGNMENT_OPERATOR()                                                                \
void operator=(Variable rhs) const noexcept {                                                              \
    Void(Function::current().add_expression(std::make_unique<BinaryExpr>(BinaryOp::ASSIGN, *this, rhs)));  \
}                                                                                                          \
template<typename U, detail::EnableIfLiteralOperand<U> = 0> void operator=(U &&rhs) const noexcept {       \
    this->operator=(literal(std::forward<U>(rhs)));                                                        \
}                                                                                                          \

template<typename T>
struct Arg : public Variable {
    Arg() noexcept: Variable{arg<T>()} {}
    MAKE_VARIABLE_ASSIGNMENT_OPERATOR()
};

template<typename T>
struct Var : public Variable {
    
    template<typename ...Literals>
    explicit Var(Literals &&...vs) noexcept : Variable{var<T>(std::forward<Literals>(vs)...)} {}
    
    MAKE_VARIABLE_ASSIGNMENT_OPERATOR()
};

using Bool = Var<bool>;
using Float = Var<float>;
using Int8 = Var<int8_t>;
using UInt8 = Var<uint8_t>;
using Int16 = Var<int16_t>;
using UInt16 = Var<uint16_t>;
using Int32 = Var<int32_t>;
using UInt32 = Var<uint32_t>;
using Int64 = Var<int64_t>;
using UInt64 = Var<uint64_t>;

struct Auto : public Variable {
    explicit Auto(LiteralExpr::Value v) noexcept: Variable{var(std::move(v))} {}
    MAKE_VARIABLE_ASSIGNMENT_OPERATOR()
};

template<typename T>
struct Let : public Variable {
    
    template<typename ...Literals>
    explicit Let(Literals &&...vs) noexcept : Variable{let(std::forward<Literals>(vs)...)} {}
    
    MAKE_VARIABLE_ASSIGNMENT_OPERATOR()
};

template<typename T>
struct LambdaArgument : public Variable {
    LambdaArgument(Variable v) noexcept: Variable{Function::current().var<T>(v)} {}
    MAKE_VARIABLE_ASSIGNMENT_OPERATOR()
};

// Used for arguments passed by value
template<typename T>
using Copy = LambdaArgument<T>;

// Used for arguments passed by value
using Ref = Variable;

#undef MAKE_VARIABLE_ASSIGNMENT_OPERATOR

template<typename True, typename False,
    std::enable_if_t<std::conjunction_v<std::is_invocable<True>, std::is_invocable<False>>, int> = 0>
inline void If(Variable cond, True &&true_branch, False &&false_branch) {
    auto &&f = Function::current();
    f.add_statement(std::make_unique<IfStmt>(cond));
    f.block(std::forward<True>(true_branch));
    f.add_statement(std::make_unique<KeywordStmt>("else"));
    f.block(std::forward<False>(false_branch));
}

template<typename True,
    std::enable_if_t<std::is_invocable_v<True>, int> = 0>
inline void If(Variable cond, True &&true_branch) {
    auto &&f = Function::current();
    f.add_statement(std::make_unique<IfStmt>(cond));
    f.block(std::forward<True>(true_branch));
}

template<typename Body,
    std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
inline void While(Variable cond, Body &&body) {
    auto &&f = Function::current();
    f.add_statement(std::make_unique<WhileStmt>(cond));
    f.block(std::forward<Body>(body));
}

template<typename Body,
    std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
inline void DoWhile(Body &&body, Variable cond) {
    auto &&f = Function::current();
    f.add_statement(std::make_unique<KeywordStmt>("do"));
    f.block(std::forward<Body>(body));
    f.add_statement(std::make_unique<WhileStmt>(cond));
    f.add_statement(std::make_unique<KeywordStmt>(";"));
}

template<typename Body,
    std::enable_if_t<std::is_invocable_v<Body, Variable>, int> = 0>
inline void Loop(Variable begin, Variable end, Variable step, Body &&body) {
    auto &&f = Function::current();
    auto i = f.var<AutoType>(begin);
    f.add_statement(std::make_unique<LoopStmt>(i, end, step));
    f.block([&] { body(i); });
}

template<typename Body>
inline void Loop(Variable begin, Variable end, Body &&body) {
    Loop(std::move(begin), std::move(end), literal(1), std::forward<Body>(body));
}

// For highlighting...
#define arg arg
#define var var
#define let let

#define If If
#define While While
#define Loop Loop
#define DoWhile DoWhile
#define Void Void

inline void Break() noexcept { Function::current().add_break(); }
inline void Continue() noexcept { Function::current().add_continue(); }
inline void Return() noexcept { Function::current().add_return(); }

#define Break    Break
#define Continue Continue
#define Return   Return

}

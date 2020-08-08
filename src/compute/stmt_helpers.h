//
// Created by Mike Smith on 2020/7/30.
//

#pragma once

#include <compute/function.h>
#include <compute/expr_helpers.h>

namespace luisa::dsl {

inline void void_(Variable v) { Function::current().add_statement(std::make_unique<ExprStmt>(v)); }

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

#define MAKE_VARIABLE_ASSIGNMENT_OPERATOR()                                                                 \
void operator=(Variable rhs) const noexcept {                                                               \
    void_(Function::current().add_expression(std::make_unique<BinaryExpr>(BinaryOp::ASSIGN, *this, rhs)));  \
}                                                                                                           \
template<typename U, detail::EnableIfLiteralOperand<U> = 0> void operator=(U &&rhs) const noexcept {        \
    this->operator=(literal(std::forward<U>(rhs)));                                                         \
}                                                                                                           \

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
using Auto = Var<AutoType>;

template<typename T>
struct Let : public Variable {
    
    template<typename ...Literals>
    explicit Let(Literals &&...vs) noexcept : Variable{let<T>(std::forward<Literals>(vs)...)} {}
    
    MAKE_VARIABLE_ASSIGNMENT_OPERATOR()
};

template<typename T>
struct LambdaArgument : public Variable {
    LambdaArgument(Variable v) noexcept: Variable{var<T>(std::move(v))} {}
    MAKE_VARIABLE_ASSIGNMENT_OPERATOR()
};

// Used for arguments passed by value
template<typename T>
using Copy = LambdaArgument<T>;

// Used for arguments passed by value
using Ref = Variable;

#undef MAKE_VARIABLE_ASSIGNMENT_OPERATOR

struct IfStmtBuilder {
    
    explicit IfStmtBuilder(Variable cond) noexcept { Function::current().add_statement(std::make_unique<IfStmt>(std::move(cond))); }
    
    template<typename True, std::enable_if_t<std::is_invocable_v<True>, int> = 0>
    const IfStmtBuilder &operator<<(True &&t) const noexcept {
        Function::current().block(std::forward<True>(t));
        return *this;
    }
    
    template<typename False, std::enable_if_t<std::is_invocable_v<False>, int> = 0>
    void operator>>(False &&f) const noexcept {
        Function::current().add_statement(std::make_unique<KeywordStmt>("else"));
        Function::current().block(std::forward<False>(f));
    }
    
    // for else if
    const IfStmtBuilder &operator>>(Variable else_cond) const noexcept {
        Function::current().add_statement(std::make_unique<KeywordStmt>("else"));
        Function::current().add_statement(std::make_unique<IfStmt>(std::move(else_cond), true));
        return *this;
    }
};

struct SwitchStmtBuilder {
    
    explicit SwitchStmtBuilder(Variable expr) noexcept { Function::current().add_statement(std::make_unique<SwitchStmt>(std::move(expr))); }
    
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    void operator<<(Body &&body) const noexcept {
        Function::current().block(std::forward<Body>(body));
    }
};

struct CaseStmtBuilder {
    
    explicit CaseStmtBuilder(Variable expr) noexcept { Function::current().add_statement(std::make_unique<CaseStmt>(std::move(expr))); }
    CaseStmtBuilder() noexcept { Function::current().add_statement(std::make_unique<CaseStmt>()); }
    
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    void operator<<(Body &&body) const noexcept {
        Function::current().block(std::forward<Body>(body));
    }
};

struct WhileStmtBuilder {
    
    explicit WhileStmtBuilder(Variable cond) noexcept { Function::current().add_statement(std::make_unique<WhileStmt>(std::move(cond))); }
    
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    void operator<<(Body &&body) const noexcept {
        Function::current().block(std::forward<Body>(body));
    }
};

struct LoopWhenStmtBuilder {
    
    LoopWhenStmtBuilder() noexcept { Function::current().add_statement(std::make_unique<KeywordStmt>("do")); }
    
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    const LoopWhenStmtBuilder &operator<<(Body &&body) const noexcept {
        Function::current().block(std::forward<Body>(body));
        return *this;
    }
    
    void operator>>(Variable cond) const noexcept {
        Function::current().add_statement(std::make_unique<WhileStmt>(std::move(cond), true));
    }
};

class ForStmtBuilder {

private:
    Variable _i;

public:
    
    template<typename Begin, typename End, typename Step>
    ForStmtBuilder(Begin &&begin, End &&end, Step &&step) noexcept
        : _i{var<AutoType>(std::forward<Begin>(begin))} {
        Function::current().add_statement(std::make_unique<ForStmt>(_i, literal(std::forward<End>(end)), literal(std::forward<Step>(step))));
    }
    
    template<typename Begin, typename End>
    ForStmtBuilder(Begin &&begin, End &&end) noexcept
        : ForStmtBuilder(std::forward<Begin>(begin), std::forward<End>(end), literal(1)) {}
    
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body, Variable>, int> = 0>
    void operator<<(Body &&body) const noexcept {
        Function::current().block([&] { body(_i); });
    }
};

// For highlighting...
#define arg arg
#define var var
#define let let

#define If(...)     IfStmtBuilder{literal(__VA_ARGS__)} << [&]
#define Else        >> [&]
#define Elif(...)   >> literal(__VA_ARGS__) << [&]

#define Switch(...) SwitchStmtBuilder{literal(__VA_ARGS__)} << [&]
#define Case(...)   CaseStmtBuilder{literal(__VA_ARGS__)} << [&]
#define Default     CaseStmtBuilder{} << [&]

#define While(...)  WhileStmtBuilder{literal(__VA_ARGS__)} << [&]

#define Loop        LoopWhenStmtBuilder{} << [&]
#define When(...)   >> literal(__VA_ARGS__)

#define For(v, ...) ForStmtBuilder{__VA_ARGS__} << [&](Variable v)

inline void break_() noexcept { Function::current().add_break(); }
inline void continue_() noexcept { Function::current().add_continue(); }
inline void return_() noexcept { Function::current().add_return(); }

#define Break    break_()
#define Continue continue_()
#define Return   return_()

}

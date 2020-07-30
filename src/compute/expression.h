//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <utility>
#include <compute/variable.h>

namespace luisa::dsl {
class Variable;
}

namespace luisa::dsl {

class Expression {

private:
    Function *_function;

public:
    virtual ~Expression() noexcept = default;
    explicit Expression(Function *func) noexcept: _function{func} {}
    [[nodiscard]] Function *function() const noexcept { return _function; }
};

enum struct UnaryOp {
    ADDRESS_OF,   // &x
    DEREFERENCE,  // *x
    PREFIX_INC,   // ++x
    PREFIX_DEC,   // --x
    POSTFIX_INC,  // x++
    POSTFIX_DEC   // x--
};

class UnaryExpr : public Expression {

private:
    Variable _operand;
    UnaryOp _op;

public:
    UnaryExpr(UnaryOp op, Variable operand) noexcept
        : Expression{operand.function()}, _operand{std::move(operand)}, _op{op} {}
    
    [[nodiscard]] Variable operand() const noexcept { return _operand; }
    [[nodiscard]] UnaryOp op() const noexcept { return _op; }
};

enum struct BinaryOp {
    
    // arithmetic
    ADD, SUB, MUL, DIV, MOD,
    BIT_AND, BIT_OR, BIT_XOR, SHL, SHR,
    AND, OR,
    
    // relational
    LESS, GREATER, LESS_EQUAL, GREATER_EQUAL, EQUAL, NOT_EQUAL,
    
    // operator[]
    ACCESS,
    
    // assignment
    ASSIGN,
    ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, MOD_ASSIGN,
    BIT_AND_ASSIGN, BIT_OR_ASSIGN, BIT_XOR_ASSIGN, SHL_ASSIGN, SHR_ASSIGN
};

class BinaryExpr : public Expression {

private:
    Variable _lhs;
    Variable _rhs;
    BinaryOp _op;

public:
    BinaryExpr(BinaryOp op, Variable lhs, Variable rhs) noexcept
        : Expression{lhs.function()}, _op{op}, _lhs{std::move(lhs)}, _rhs{std::move(rhs)} {}
    
    [[nodiscard]] Variable lhs() const noexcept { return _lhs; }
    [[nodiscard]] Variable rhs() const noexcept { return _rhs; }
    [[nodiscard]] BinaryOp op() const noexcept { return _op; }
};

class MemberExpr : public Expression {

private:
    Variable _self;
    std::string _member;

public:
    MemberExpr(Variable self, std::string member) noexcept: Expression{self.function()}, _self{self}, _member{std::move(member)} {}
    [[nodiscard]] Variable self() const noexcept { return _self; }
    [[nodiscard]] const std::string &member() const noexcept { return _member; }
};

class ArrowExpr : public Expression {

private:
    Variable _self;
    std::string _member;

public:
    ArrowExpr(Variable self, std::string member) noexcept: Expression{self.function()}, _self{self}, _member{std::move(member)} {}
    [[nodiscard]] Variable self() const noexcept { return _self; }
    [[nodiscard]] const std::string &member() const noexcept { return _member; }
};

}

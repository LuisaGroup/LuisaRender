//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <compute/variable.h>

namespace luisa::dsl {
class Variable;
}

namespace luisa::dsl {

class Expression {

private:
    Function *_function;

public:
    explicit Expression(Function *func) noexcept: _function{func} {}
    [[nodiscard]] Function *function() const noexcept { return _function; }
};

enum struct BinaryOp {
    ADD, SUB, MUL, DIV, MOD,                                     // arithmetic
    BIT_AND, BIT_OR, BIT_XOR, SHL, SHR,                          // bit-wise
    AND, OR,                                                     // logical
    LESS, GREATER, LESS_EQUAL, GREATER_EQUAL, EQUAL, NOT_EQUAL,  // relational
    ACCESS,                                                      // access
};

class BinaryExpr : public Expression {

private:
    Variable _lhs;
    Variable _rhs;
    BinaryOp _op;

public:
    BinaryExpr(BinaryOp op, Variable lhs, Variable rhs) noexcept
        : Expression{lhs.function()}, _op{op}, _lhs{lhs}, _rhs{rhs} {}
    
    [[nodiscard]] Variable lhs() const noexcept { return _lhs; }
    [[nodiscard]] Variable rhs() const noexcept { return _rhs; }
    [[nodiscard]] BinaryOp op() const noexcept { return _op; }
};

}

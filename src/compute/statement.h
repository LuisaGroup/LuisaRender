//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <functional>
#include <utility>

#include <compute/variable.h>
#include <compute/expression.h>

namespace luisa::dsl {

class Statement {

};

class DeclareStmt : public Statement {

private:
    Variable _var;
    Expression *_init_expr;

public:
    DeclareStmt(Variable var, Expression *init) noexcept: _var{std::move(var)}, _init_expr{init} {}
    [[nodiscard]] Variable var() const noexcept { return _var; }
    [[nodiscard]] Expression *init_expr() const noexcept { return _init_expr; }
};

class BlockBeginStmt : public Statement {};

class BlockEndStmt : public Statement {};

class IfStmt : public Statement {

private:
    Expression *_condition;

public:
    explicit IfStmt(Expression *cond) noexcept: _condition{cond} {}
    [[nodiscard]] Expression *condition() const noexcept { return _condition; }
};

class ElseStmt : public Statement {

};

void if_(Variable cond, const std::function<void()> &true_branch);
void if_(Variable cond, const std::function<void()> &true_branch, const std::function<void()> &false_branch);

enum struct AssignOp {
    ASSIGN,
    ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, MOD_ASSIGN,             // arithmetic
    BIT_AND_ASSIGN, BIT_OR_ASSIGN, BIT_XOR_ASSIGN, SHL_ASSIGN, SHR_ASSIGN,  // bit-wise
};

class AssignStmt : public Statement {

private:
    Variable _lvalue;
    AssignOp _op;
    Expression *_expr;

public:
    AssignStmt(AssignOp op, Variable lvalue, Expression *expr) noexcept : _lvalue{std::move(lvalue)}, _op{op}, _expr{expr} {}
    [[nodiscard]] Variable lvalue() const noexcept { return _lvalue; }
    [[nodiscard]] Expression *expression() const noexcept { return _expr; }
    [[nodiscard]] AssignOp op() const noexcept { return _op; }
};

}

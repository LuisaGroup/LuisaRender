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

class ScopeBeginStmt : public Statement {

};

class ScopeEndStmt : public Statement {

};

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

class AssignStmt : public Statement {

private:
    Variable _lvalue;
    Expression *_expr;

public:
    AssignStmt(Variable lvalue, Expression *expr) noexcept : _lvalue{std::move(lvalue)}, _expr{expr} {}
    [[nodiscard]] Variable lvalue() const noexcept { return _lvalue; }
    [[nodiscard]] Expression *expression() const noexcept { return _expr; }
};

}

//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <functional>
#include <utility>

#include <compute/variable.h>
#include <compute/expression.h>

namespace luisa::dsl {

// fwd-decl
struct StmtVisitor;

// Statement interface
struct Statement {
    virtual ~Statement() noexcept = default;
    virtual void accept(const StmtVisitor &visitor) const = 0;
};

// fwd-decl of derived statments
class DeclareStmt;
struct BlockBeginStmt;
struct BlockEndStmt;
class IfStmt;
struct ElseStmt;
class ExprStmt;

// Statement visitor interface
struct StmtVisitor {
    virtual void visit(const DeclareStmt &declare_stmt) const = 0;
    virtual void visit(const BlockBeginStmt &block_begin_stmt) const = 0;
    virtual void visit(const BlockEndStmt &block_end_stmt) const = 0;
    virtual void visit(const IfStmt &if_stmt) const = 0;
    virtual void visit(const ElseStmt &else_stmt) const = 0;
    virtual void visit(const ExprStmt &expr_stmt) const = 0;
};

#define MAKE_STATEMENT_ACCEPT_VISITOR()                                           \
void accept(const StmtVisitor &visitor) const override { visitor.visit(*this); }  \

class DeclareStmt : public Statement {

private:
    Variable _var;
    Expression *_init_expr;

public:
    DeclareStmt(Variable var, Expression *init) noexcept: _var{std::move(var)}, _init_expr{init} {}
    [[nodiscard]] Variable var() const noexcept { return _var; }
    [[nodiscard]] Expression *init_expr() const noexcept { return _init_expr; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct BlockBeginStmt : public Statement {
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct BlockEndStmt : public Statement {
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class IfStmt : public Statement {

private:
    Expression *_condition;

public:
    explicit IfStmt(Expression *cond) noexcept: _condition{cond} {}
    [[nodiscard]] Expression *condition() const noexcept { return _condition; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct ElseStmt : public Statement {
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

void if_(Variable cond, const std::function<void()> &true_branch);
void if_(Variable cond, const std::function<void()> &true_branch, const std::function<void()> &false_branch);

class ExprStmt : public Statement {

private:
    Expression *_expr;

public:
    explicit ExprStmt(Expression *expr) noexcept : _expr{expr} {}
    [[nodiscard]] Expression *expression() const noexcept { return _expr; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

void void_(Variable v);

#undef MAKE_STATEMENT_ACCEPT_VISITOR

}

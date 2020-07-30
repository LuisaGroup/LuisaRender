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
    virtual void accept(StmtVisitor &visitor) const = 0;
};

// fwd-decl of derived statments
class DeclareStmt;
class KeywordStmt;
class IfStmt;
class WhileStmt;
class LoopStmt;
class ExprStmt;

// Statement visitor interface
struct StmtVisitor {
    virtual void visit(const DeclareStmt &declare_stmt) = 0;
    virtual void visit(const KeywordStmt &stmt) = 0;
    virtual void visit(const IfStmt &if_stmt) = 0;
    virtual void visit(const WhileStmt &while_stmt) = 0;
    virtual void visit(const LoopStmt &loop_stmt) = 0;
    virtual void visit(const ExprStmt &expr_stmt) = 0;
};

#define MAKE_STATEMENT_ACCEPT_VISITOR()                                     \
void accept(StmtVisitor &visitor) const override { visitor.visit(*this); }  \

class DeclareStmt : public Statement {

private:
    Variable _var;
    Variable _init;

public:
    DeclareStmt(Variable var, Variable init) noexcept: _var{std::move(var)}, _init{std::move(init)} {}
    [[nodiscard]] Variable var() const noexcept { return _var; }
    [[nodiscard]] Variable initialization() const noexcept { return _init; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class IfStmt : public Statement {

private:
    Variable _condition;

public:
    explicit IfStmt(Variable cond) noexcept: _condition{std::move(cond)} {}
    [[nodiscard]] Variable condition() const noexcept { return _condition; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class WhileStmt : public Statement {

private:
    Variable _condition;

public:
    explicit WhileStmt(Variable cond) noexcept: _condition{std::move(cond)} {}
    [[nodiscard]] Variable condition() const noexcept { return _condition; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class LoopStmt : public Statement {

private:
    Variable _i;
    Variable _end;
    Variable _step;

public:
    LoopStmt(Variable i, Variable end, Variable step) noexcept
        : _i{std::move(i)}, _end{std::move(end)}, _step{std::move(step)} {}
    [[nodiscard]] Variable i() const noexcept { return _i; }
    [[nodiscard]] Variable end() const noexcept { return _end; }
    [[nodiscard]] Variable step() const noexcept { return _step; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class KeywordStmt : public Statement {

private:
    std::string_view _keyword;

public:
    explicit KeywordStmt(std::string_view keyword) noexcept: _keyword{keyword} {}
    [[nodiscard]] std::string_view keyword() const noexcept { return _keyword; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ExprStmt : public Statement {

private:
    Variable _expr;

public:
    explicit ExprStmt(Variable expr) noexcept: _expr{std::move(expr)} {}
    [[nodiscard]] Variable expression() const noexcept { return _expr; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

#undef MAKE_STATEMENT_ACCEPT_VISITOR

}

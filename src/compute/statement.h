//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <functional>
#include <utility>
#include <optional>

#include <compute/variable.h>
#include <compute/expression.h>

namespace luisa::compute::dsl {

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
class ForStmt;
class ExprStmt;
class SwitchStmt;
class CaseStmt;

// Statement visitor interface
struct StmtVisitor {
    virtual void visit(const DeclareStmt &declare_stmt) = 0;
    virtual void visit(const KeywordStmt &stmt) = 0;
    virtual void visit(const IfStmt &if_stmt) = 0;
    virtual void visit(const WhileStmt &while_stmt) = 0;
    virtual void visit(const ForStmt &loop_stmt) = 0;
    virtual void visit(const ExprStmt &expr_stmt) = 0;
    virtual void visit(const SwitchStmt &switch_stmt) = 0;
    virtual void visit(const CaseStmt &case_stmt) = 0;
};

#define MAKE_STATEMENT_ACCEPT_VISITOR()                                     \
void accept(StmtVisitor &visitor) const override { visitor.visit(*this); }  \

class DeclareStmt : public Statement {

private:
    Variable _var;
    Variable _init;
    bool _is_constant;

public:
    DeclareStmt(Variable var, Variable init, bool is_const) noexcept
        : _var{std::move(var)}, _init{std::move(init)}, _is_constant{is_const} {}
    [[nodiscard]] Variable var() const noexcept { return _var; }
    [[nodiscard]] Variable initialization() const noexcept { return _init; }
    [[nodiscard]] bool is_constant() const noexcept { return _is_constant; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class IfStmt : public Statement {

private:
    Variable _condition;
    bool _is_elif;

public:
    explicit IfStmt(Variable cond, bool is_elif = false) noexcept: _condition{std::move(cond)}, _is_elif{is_elif} {}
    [[nodiscard]] Variable condition() const noexcept { return _condition; }
    [[nodiscard]] bool is_elif() const noexcept { return _is_elif; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchStmt : public Statement {

private:
    Variable _expr;

public:
    explicit SwitchStmt(Variable expr) noexcept : _expr{std::move(expr)} {}
    [[nodiscard]] Variable expression() const noexcept { return _expr; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class CaseStmt : public Statement {

private:
    Variable _expr;

public:
    CaseStmt() noexcept = default;
    explicit CaseStmt(Variable expr) noexcept : _expr{std::move(expr)} {}
    [[nodiscard]] Variable expression() const noexcept { return _expr; }
    [[nodiscard]] bool is_default() const noexcept { return !_expr.is_valid(); }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class WhileStmt : public Statement {

private:
    Variable _condition;
    bool _is_do_while;

public:
    explicit WhileStmt(Variable cond, bool is_do_while = false) noexcept
        : _condition{std::move(cond)}, _is_do_while{is_do_while} {}
    [[nodiscard]] Variable condition() const noexcept { return _condition; }
    [[nodiscard]] bool is_do_while() const noexcept { return _is_do_while; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ForStmt : public Statement {

private:
    Variable _i;
    Variable _end;
    Variable _step;

public:
    ForStmt(Variable i, Variable end, Variable step) noexcept
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

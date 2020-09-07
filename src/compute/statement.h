//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <functional>
#include <utility>
#include <optional>

#include <compute/variable.h>
#include <compute/expression.h>
#include <compute/function.h>

namespace luisa::compute::dsl {

// fwd-decl
struct StmtVisitor;

// Statement interface
struct Statement : Noncopyable {
    virtual ~Statement() noexcept = default;
    virtual void accept(StmtVisitor &visitor) const = 0;
};

// fwd-decl of derived statments
struct EmptyStmt;

struct BreakStmt;
struct ContinueStmt;
struct ReturnStmt;

class ScopeStmt;
class DeclareStmt;
class IfStmt;
class WhileStmt;
class DoWhileStmt;
class ExprStmt;
class SwitchStmt;
class SwitchCaseStmt;
class SwitchDefaultStmt;

// Statement visitor interface
struct StmtVisitor {
    virtual void visit(const EmptyStmt *) = 0;
    virtual void visit(const BreakStmt *) = 0;
    virtual void visit(const ContinueStmt *) = 0;
    virtual void visit(const ReturnStmt *) = 0;
    virtual void visit(const ScopeStmt *scope_stmt) = 0;
    virtual void visit(const DeclareStmt *declare_stmt) = 0;
    virtual void visit(const IfStmt *if_stmt) = 0;
    virtual void visit(const WhileStmt *while_stmt) = 0;
    virtual void visit(const DoWhileStmt *do_while_stmt) = 0;
    virtual void visit(const ExprStmt *expr_stmt) = 0;
    virtual void visit(const SwitchStmt *switch_stmt) = 0;
    virtual void visit(const SwitchCaseStmt *case_stmt) = 0;
    virtual void visit(const SwitchDefaultStmt *default_stmt) = 0;
};

#define MAKE_STATEMENT_ACCEPT_VISITOR()                                    \
void accept(StmtVisitor &visitor) const override { visitor.visit(this); }  \

struct EmptyStmt : public Statement {
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct BreakStmt : public Statement {
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct ContinueStmt : public Statement {
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

struct ReturnStmt : public Statement {
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ScopeStmt : public Statement {

private:
    std::vector<std::unique_ptr<Statement>> _statements;

public:
    void add_statement(std::unique_ptr<Statement> stmt) noexcept { _statements.emplace_back(std::move(stmt)).get(); }
    [[nodiscard]] const std::vector<std::unique_ptr<Statement>> &statements() const noexcept { return _statements; }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class DeclareStmt : public Statement {

private:
    const Variable *_var;
    const Expression *_init_expr;
    bool _is_constexpr;

public:
    DeclareStmt(const Variable *var, const Expression *init, bool is_constexpr) noexcept
        : _var{var}, _init_expr{init}, _is_constexpr{is_constexpr} {}
    [[nodiscard]] const Variable *var() const noexcept { return _var; }
    [[nodiscard]] const Expression *init_expr() const noexcept { return _init_expr; }
    [[nodiscard]] bool is_constexpr() const noexcept { return _is_constexpr; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class IfStmt : public Statement {

private:
    const Expression *_condition;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    IfStmt(const Expression *cond, Body &&body) : _condition{cond}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), std::forward<Body>(body));
    }
    [[nodiscard]] const Expression *condition() const noexcept { return _condition; }
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchCaseStmt : public Statement {

private:
    const Expression *_expr;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    SwitchCaseStmt(const Expression *expr, Body &&body) : _expr{expr}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), [&body] {
            body();
            Function::current().add_statement(std::make_unique<BreakStmt>());
        });
    }
    
    [[nodiscard]] const Expression *expr() const noexcept { return _expr; }
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchDefaultStmt : public Statement {

private:
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    explicit SwitchDefaultStmt(Body &&body) : _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), [&body] {
            body();
            Function::current().add_statement(std::make_unique<BreakStmt>());
        });
    }
    
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchStmt : public Statement {

private:
    const Expression *_expr;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    SwitchStmt(const Expression *expr, Body &&body) noexcept: _expr{expr}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), std::forward<Body>(body));
    }
    [[nodiscard]] const Expression *expr() const noexcept { return _expr; }
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class WhileStmt : public Statement {

private:
    const Expression *_condition;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    WhileStmt(const Expression *cond, Body &&body) : _condition{cond}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), std::forward<Body>(body));
    }
    [[nodiscard]] const Expression *condition() const noexcept { return _condition; }
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class DoWhileStmt : public Statement {

private:
    const Expression *_condition;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    DoWhileStmt(Body &&body, const Expression *cond) : _condition{cond}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), std::forward<Body>(body));
    }
    [[nodiscard]] const Expression *condition() const noexcept { return _condition; }
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ExprStmt : public Statement {

private:
    const Expression *_expr;

public:
    explicit ExprStmt(const Expression *expr) noexcept: _expr{expr} {}
    [[nodiscard]] const Expression *expr() const noexcept { return _expr; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

#undef MAKE_STATEMENT_ACCEPT_VISITOR

}

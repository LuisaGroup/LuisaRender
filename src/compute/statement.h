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
class AssignStmt;

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
    virtual void visit(const AssignStmt *assign_stmt) = 0;
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
    std::vector<const Variable *> _initializer_list;

public:
    DeclareStmt(const Variable *var, std::vector<const Variable *> init) noexcept
        : _var{var}, _initializer_list{std::move(init)} {}
    [[nodiscard]] const Variable *var() const noexcept { return _var; }
    [[nodiscard]] const std::vector<const Variable *> &init_expr() const noexcept { return _initializer_list; }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

enum struct AssignOp {
    ASSIGN,
    ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, MOD_ASSIGN,
    BIT_AND_ASSIGN, BIT_OR_ASSIGN, BIT_XOR_ASSIGN, SHL_ASSIGN, SHR_ASSIGN
};

class AssignStmt : public Statement {

private:
    const Variable *_lhs;
    const Variable *_rhs;
    AssignOp _op;

public:
    AssignStmt(AssignOp op, const Variable *lhs, const Variable *rhs) noexcept
        : _lhs{lhs}, _rhs{rhs}, _op{op} {}
    
    [[nodiscard]] const Variable *lhs() const noexcept { return _lhs; }
    [[nodiscard]] const Variable *rhs() const noexcept { return _rhs; }
    [[nodiscard]] AssignOp op() const noexcept { return _op; }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class IfStmt : public Statement {

private:
    const Variable *_condition;
    std::unique_ptr<ScopeStmt> _true_branch;
    std::unique_ptr<ScopeStmt> _false_branch;

public:
    template<typename True, std::enable_if_t<std::is_invocable_v<True>, int> = 0>
    IfStmt(const Variable *cond, True &&true_branch) noexcept
        : _condition{cond},
          _true_branch{std::make_unique<ScopeStmt>()},
          _false_branch{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_true_branch.get(), std::forward<True>(true_branch));
    }
    
    template<typename True, typename False, std::enable_if_t<std::conjunction_v<std::is_invocable<True>, std::is_invocable<False>>, int> = 0>
    IfStmt(const Variable *cond, True &&true_branch, False &&false_branch) noexcept
        : _condition{cond},
          _true_branch{std::make_unique<ScopeStmt>()},
          _false_branch{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_true_branch.get(), std::forward<True>(true_branch));
        Function::current().with_scope(_false_branch.get(), std::forward<False>(false_branch));
    }
    
    [[nodiscard]] const Variable *condition() const noexcept { return _condition; }
    [[nodiscard]] const ScopeStmt *true_branch() const noexcept { return _true_branch.get(); }
    [[nodiscard]] const ScopeStmt *false_branch() const noexcept { return _false_branch.get(); }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class SwitchCaseStmt : public Statement {

private:
    const Variable *_expr;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    SwitchCaseStmt(const Variable *expr, Body &&body) : _expr{expr}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), [&body] {
            body();
            Function::current().add_statement(std::make_unique<BreakStmt>());
        });
    }
    
    [[nodiscard]] const Variable *expr() const noexcept { return _expr; }
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
    const Variable *_expr;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    SwitchStmt(const Variable *expr, Body &&body) noexcept: _expr{expr}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), std::forward<Body>(body));
    }
    [[nodiscard]] const Variable *expr() const noexcept { return _expr; }
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class WhileStmt : public Statement {

private:
    const Variable *_condition;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    WhileStmt(const Variable *cond, Body &&body) : _condition{cond}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), std::forward<Body>(body));
    }
    [[nodiscard]] const Variable *condition() const noexcept { return _condition; }
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class DoWhileStmt : public Statement {

private:
    const Variable *_condition;
    std::unique_ptr<ScopeStmt> _body;

public:
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    DoWhileStmt(Body &&body, const Variable *cond) : _condition{cond}, _body{std::make_unique<ScopeStmt>()} {
        Function::current().with_scope(_body.get(), std::forward<Body>(body));
    }
    [[nodiscard]] const Variable *condition() const noexcept { return _condition; }
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

class ExprStmt : public Statement {

private:
    std::unique_ptr<Expression> _expr;

public:
    explicit ExprStmt(std::unique_ptr<Expression> expr) noexcept: _expr{std::move(expr)} {}
    [[nodiscard]] const Expression *expr() const noexcept { return _expr.get(); }
    MAKE_STATEMENT_ACCEPT_VISITOR()
};

#undef MAKE_STATEMENT_ACCEPT_VISITOR

}

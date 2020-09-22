//
// Created by Mike on 9/22/2020.
//

#pragma once

#include "dsl.h"

#define Return ::luisa::compute::dsl::Function::current().add_return()
#define Break ::luisa::compute::dsl::Function::current().add_break()
#define Continue ::luisa::compute::dsl::Function::current().add_continue()

namespace luisa::compute::dsl {

class IfStmtBuilder {

private:
    const Variable *_cond{nullptr};
    std::function<void()> _true;
    std::function<void()> _false;

public:
    template<typename Cond>
    explicit IfStmtBuilder(Cond &&cond) noexcept {
        Expr cond_expr{std::forward<Cond>(cond)};
        _cond = cond_expr.variable();
    }
    
    ~IfStmtBuilder() noexcept {
        if (_false) {
            Function::current().add_statement(std::make_unique<IfStmt>(_cond, std::move(_true), std::move(_false)));
        } else {
            Function::current().add_statement(std::make_unique<IfStmt>(_cond, std::move(_true)));
        }
    }
    
    IfStmtBuilder &operator<<(std::function<void()> t) noexcept {
        _true = std::move(t);
        return *this;
    }
    
    void operator>>(std::function<void()> f) noexcept { _false = std::move(f); }
    
    [[nodiscard]] IfStmtBuilder &operator<<(IfStmtBuilder *elif) noexcept {
        _false = [elif] { delete elif; };
        return *elif;
    }
};

#define If(...) ::luisa::compute::dsl::IfStmtBuilder{__VA_ARGS__} << [&]()
#define Else >> [&]()
#define Elif(...) << (new ::luisa::compute::dsl::IfStmtBuilder{__VA_ARGS__}) << [&]()

class WhileStmtBuilder {

private:
    std::function<Expr<bool>()> _cond;
    std::function<void()> _body;

public:
    template<typename Cond>
    explicit WhileStmtBuilder(Cond &&cond) noexcept
        : _cond{std::forward<Cond>(cond)} {}
    
    ~WhileStmtBuilder() noexcept {
        Expr always_true{true};
        Function::current().add_statement(std::make_unique<WhileStmt>(always_true.variable(), std::move(_body)));
    }
    
    template<typename Body, std::enable_if_t<std::is_invocable_v<Body>, int> = 0>
    void operator<<(Body &&body) noexcept {
        _body = [this, &body] {
            If (!_cond()) { Break; };  // Work-around for local Var decls in cond
            body();
        };
    }
};

#define While(...) ::luisa::compute::dsl::WhileStmtBuilder{[&]{ return __VA_ARGS__; }} << [&]()

class SwitchStmtBuilder {

private:
    const Variable *_expr{nullptr};

public:
    template<typename T>
    explicit SwitchStmtBuilder(T &&expr) noexcept {
        Expr e{expr};
        _expr = e.variable();
    }
    
    template<typename F>
    void operator<<(F &&f) noexcept {
        Function::current().add_statement(std::make_unique<SwitchStmt>(_expr, std::forward<F>(f)));
    }
};

class SwitchCaseStmtBuilder {

private:
    const Variable *_expr{nullptr};

public:
    template<typename T>
    explicit SwitchCaseStmtBuilder(T &&expr) noexcept {
        Expr e{expr};
        _expr = e.variable();
    }
    
    template<typename F>
    void operator<<(F &&f) noexcept {
        Function::current().add_statement(std::make_unique<SwitchCaseStmt>(_expr, std::forward<F>(f)));
    }
};

struct SwitchDefaultStmtBuilder {
    template<typename F>
    void operator<<(F &&f) noexcept {
        Function::current().add_statement(std::make_unique<SwitchDefaultStmt>(std::forward<F>(f)));
    }
};

#define Switch(...) ::luisa::compute::dsl::SwitchStmtBuilder{__VA_ARGS__} << [&]()
#define Case(...) ::luisa::compute::dsl::SwitchCaseStmtBuilder{__VA_ARGS__} << [&]()
#define Default ::luisa::compute::dsl::SwitchDefaultStmtBuilder{} << [&]()

}

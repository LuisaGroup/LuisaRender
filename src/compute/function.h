//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <memory>
#include <vector>
#include <set>
#include <map>

#include <core/platform.h>

namespace luisa::compute::dsl {

class Variable;
struct Statement;
class ScopeStmt;

class Function {

private:
    std::string _name;
    std::vector<std::unique_ptr<Variable>> _builtins;
    std::vector<std::unique_ptr<Variable>> _variables;
    std::vector<std::unique_ptr<Variable>> _arguments;
    
    std::unique_ptr<ScopeStmt> _body;
    std::stack<ScopeStmt *> _scope_stack;

public:
    explicit Function(std::string name) noexcept;
    ~Function() noexcept;
    
    [[nodiscard]] static Function &current() noexcept;
    [[nodiscard]] const std::string &name() const noexcept { return _name; }
    
    template<typename F, std::enable_if_t<std::is_invocable_v<F>, int> = 0>
    void with_scope(ScopeStmt *scope, F &&f) {
        _scope_stack.push(scope);
        f();
        _scope_stack.pop();
    }
    
    void add_statement(std::unique_ptr<Statement> stmt) noexcept;
    
    [[nodiscard]] const std::vector<std::unique_ptr<Variable>> &builtins() const noexcept { return _builtins; }
    [[nodiscard]] const std::vector<std::unique_ptr<Variable>> &variables() const noexcept { return _variables; }
    [[nodiscard]] const std::vector<std::unique_ptr<Variable>> &arguments() const noexcept { return _arguments; }
    
    [[nodiscard]] const std::vector<std::unique_ptr<Statement>> &statements() const noexcept;
    
    // new api
    [[nodiscard]] const Variable *add_builtin(std::unique_ptr<Variable> v) noexcept { return _builtins.emplace_back(std::move(v)).get(); }
    [[nodiscard]] const Variable *add_variable(std::unique_ptr<Variable> v) noexcept { return _variables.emplace_back(std::move(v)).get(); }
    [[nodiscard]] const Variable *add_argument(std::unique_ptr<Variable> v) noexcept { return _arguments.emplace_back(std::move(v)).get(); }
};

}

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

public:
    static constexpr auto texture_read_bit = 1u;
    static constexpr auto texture_write_bit = 2u;
    static constexpr auto texture_sample_bit = 4u;

private:
    std::string _name;
    std::vector<std::unique_ptr<Variable>> _builtins;
    std::vector<std::unique_ptr<Variable>> _variables;
    std::vector<std::unique_ptr<Variable>> _arguments;
    std::map<const Texture *, uint32_t> _texture_usages;
    
    std::unique_ptr<ScopeStmt> _body;
    std::stack<ScopeStmt *> _scope_stack;
    
    uint32_t _uid_counter{0u};

public:
    explicit Function(std::string name) noexcept;
    ~Function() noexcept;
    
    [[nodiscard]] static Function &current() noexcept;
    [[nodiscard]] uint32_t next_uid() noexcept { return ++_uid_counter; }
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
    
    void mark_texture_read(const Texture *texture) noexcept {
        if (auto iter = _texture_usages.find(texture); iter == _texture_usages.cend()) { _texture_usages.emplace(texture, texture_read_bit); }
        else { iter->second |= texture_read_bit; }
    }
    
    void mark_texture_write(const Texture *texture) noexcept {
        if (auto iter = _texture_usages.find(texture); iter == _texture_usages.cend()) { _texture_usages.emplace(texture, texture_write_bit); }
        else { iter->second |= texture_write_bit; }
    }
    
    void mark_texture_sample(const Texture *texture) noexcept {
        if (auto iter = _texture_usages.find(texture); iter == _texture_usages.cend()) { _texture_usages.emplace(texture, texture_sample_bit); }
        else { iter->second |= texture_sample_bit; }
    }
    
    [[nodiscard]] uint32_t texture_usage(const Texture *texture) const noexcept {
        auto iter = _texture_usages.find(texture);
        if (iter == _texture_usages.cend()) { return 0u; }
        return iter->second;
    }
};

}

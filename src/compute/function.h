//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <memory>
#include <vector>
#include <set>
#include <map>
#include <stack>

#include <core/platform.h>

namespace luisa::compute {
class Texture;
}

namespace luisa::compute::dsl {

struct TypeDesc;
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
    std::vector<std::unique_ptr<Variable>> _threadgroup_variables;
    std::vector<std::unique_ptr<Variable>> _arguments;
    std::map<const Texture *, uint32_t> _texture_usages;
    std::vector<const TypeDesc *> _used_struct_types;
    
    std::unique_ptr<ScopeStmt> _body;
    std::stack<ScopeStmt *> _scope_stack;
    
    uint32_t _uid_counter{0u};
    
    void _use_structure_type(const TypeDesc *type) noexcept;

public:
    explicit Function(std::string name) noexcept;
    [[nodiscard]] static Function &current() noexcept;
    static void push(Function *f) noexcept;
    static void pop(Function *f) noexcept;
    
    [[nodiscard]] uint32_t next_uid() noexcept { return ++_uid_counter; }
    [[nodiscard]] const std::string &name() const noexcept { return _name; }
    
    template<typename F, std::enable_if_t<std::is_invocable_v<F>, int> = 0>
    void with_scope(ScopeStmt *scope, F &&f) {
        _scope_stack.push(scope);
        f();
        _scope_stack.pop();
    }
    
    void add_return();
    void add_break();
    void add_continue();
    
    void add_statement(std::unique_ptr<Statement> stmt) noexcept;
    
    [[nodiscard]] const std::vector<const TypeDesc *> &used_structures() const noexcept { return _used_struct_types; }
    [[nodiscard]] const std::vector<std::unique_ptr<Variable>> &builtins() const noexcept { return _builtins; }
    [[nodiscard]] const std::vector<std::unique_ptr<Variable>> &variables() const noexcept { return _variables; }
    [[nodiscard]] const std::vector<std::unique_ptr<Variable>> &threadgroup_variables() const noexcept { return _threadgroup_variables; }
    [[nodiscard]] const std::vector<std::unique_ptr<Variable>> &arguments() const noexcept { return _arguments; }
    
    [[nodiscard]] const ScopeStmt *body() const noexcept { return _body.get(); }
    
    // new api
    [[nodiscard]] const Variable *add_builtin(std::unique_ptr<Variable> v) noexcept;
    [[nodiscard]] const Variable *add_variable(std::unique_ptr<Variable> v) noexcept;
    [[nodiscard]] const Variable *add_threadgroup_variable(std::unique_ptr<Variable> v) noexcept;
    [[nodiscard]] const Variable *add_argument(std::unique_ptr<Variable> v) noexcept;
    
    void mark_texture_read(const Texture *texture) noexcept;
    void mark_texture_write(const Texture *texture) noexcept;
    void mark_texture_sample(const Texture *texture) noexcept;
    [[nodiscard]] uint32_t texture_usage(const Texture *texture) const noexcept;
};

}

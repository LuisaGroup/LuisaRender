//
// Created by Mike on 8/27/2020.
//

#include <compute/type_desc.h>
#include <compute/statement.h>
#include "function.h"

namespace luisa::compute::dsl {

static thread_local std::stack<Function *> _current;

Function &Function::current() noexcept {
    assert(!_current.empty());
    return *_current.top();
}

Function::Function(std::string name) noexcept: _name{std::move(name)} {
    _body = std::make_unique<ScopeStmt>();
    _scope_stack.push(_body.get());
}

void Function::add_statement(std::unique_ptr<Statement> stmt) noexcept {
    assert(!_scope_stack.empty());
    auto scope = _scope_stack.top();
    scope->add_statement(std::move(stmt));
}

void Function::_use_structure_type(const TypeDesc *type) noexcept {
    if (type == nullptr || type->type == TypeCatalog::UNKNOWN) { return; }
    if (type->type == TypeCatalog::ARRAY) {
        _use_structure_type(type->element_type);
    } else if (type->type == TypeCatalog::STRUCTURE &&
               std::none_of(_used_struct_types.cbegin(), _used_struct_types.cend(), [type](const TypeDesc *td) noexcept {
                   return td == type || td->uid() == type->uid() || td->identifier == type->identifier;
               })) {
        _used_struct_types.emplace_back(type);
        for (auto member_type : type->member_types) { _use_structure_type(member_type); }
    }
}

const Variable *Function::add_builtin(std::unique_ptr<Variable> v) noexcept {
    _use_structure_type(v->type());
    return _builtins.emplace_back(std::move(v)).get();
}

const Variable *Function::add_variable(std::unique_ptr<Variable> v) noexcept {
    _use_structure_type(v->type());
    return _variables.emplace_back(std::move(v)).get();
}

const Variable *Function::add_argument(std::unique_ptr<Variable> v) noexcept {
    _use_structure_type(v->type());
    return _arguments.emplace_back(std::move(v)).get();
}

void Function::mark_texture_read(const Texture *texture) noexcept {
    if (auto iter = _texture_usages.find(texture); iter == _texture_usages.cend()) { _texture_usages.emplace(texture, texture_read_bit); }
    else { iter->second |= texture_read_bit; }
}

void Function::mark_texture_write(const Texture *texture) noexcept {
    if (auto iter = _texture_usages.find(texture); iter == _texture_usages.cend()) { _texture_usages.emplace(texture, texture_write_bit); }
    else { iter->second |= texture_write_bit; }
}

void Function::mark_texture_sample(const Texture *texture) noexcept {
    if (auto iter = _texture_usages.find(texture); iter == _texture_usages.cend()) { _texture_usages.emplace(texture, texture_sample_bit); }
    else { iter->second |= texture_sample_bit; }
}

uint32_t Function::texture_usage(const Texture *texture) const noexcept {
    auto iter = _texture_usages.find(texture);
    if (iter == _texture_usages.cend()) { return 0u; }
    return iter->second;
}

const Variable *Function::add_threadgroup_variable(std::unique_ptr<Variable> v) noexcept {
    _use_structure_type(v->type());
    return _threadgroup_variables.emplace_back(std::move(v)).get();
}

void Function::add_return() { add_statement(std::make_unique<ReturnStmt>()); }
void Function::add_break() { add_statement(std::make_unique<BreakStmt>()); }
void Function::add_continue() { add_statement(std::make_unique<ContinueStmt>()); }

void Function::push(Function *f) noexcept {
    _current.push(f);
}

void Function::pop(Function *f) noexcept {
    assert(!_current.empty() && _current.top() == f);
    _current.pop();
}

}

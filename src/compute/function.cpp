//
// Created by Mike on 8/27/2020.
//

#include <compute/type_desc.h>
#include <compute/statement.h>
#include "function.h"

namespace luisa::compute::dsl {

static thread_local Function *_current = nullptr;

Function &Function::current() noexcept {
    assert(_current != nullptr);
    return *_current;
}

Function::~Function() noexcept {
    _current = nullptr;
}

Function::Function(std::string name) noexcept : _name{std::move(name)} {
    assert(_current == nullptr);
    _current = this;
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
    } else if (type->type == TypeCatalog::STRUCTURE && _used_struct_types.find(type) == _used_struct_types.cend()) {
        _used_struct_types.emplace(type);
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

}

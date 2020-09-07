//
// Created by Mike on 8/27/2020.
//

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

const std::vector<std::unique_ptr<Statement>> &Function::statements() const noexcept {
    return _body->statements();
}

}

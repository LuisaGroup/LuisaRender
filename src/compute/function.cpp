//
// Created by Mike on 8/27/2020.
//

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
}

}

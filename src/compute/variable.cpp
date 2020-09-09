//
// Created by Mike Smith on 2020/7/10.
//

#include <compute/expression.h>
#include <compute/function.h>
#include <compute/statement.h>

#include "variable.h"

namespace luisa::compute::dsl {

const Variable *Variable::make_local_variable(const TypeDesc *type) noexcept {
    auto v = std::make_unique<Variable>();
    v->_type = type;
    v->_tag = VariableTag::LOCAL;
    v->_uid = Function::current().next_uid();
    return Function::current().add_variable(std::move(v));
}

const Variable *Variable::make_threadgroup_variable(const TypeDesc *type) noexcept {
    auto v = std::make_unique<Variable>();
    v->_type = type;
    v->_tag = VariableTag::THREADGROUP;
    v->_uid = Function::current().next_uid();
    return Function::current().add_variable(std::move(v));
}

const Variable *Variable::make_uniform_argument(const TypeDesc *type, const void *data_ref) noexcept {
    for (auto &&v : Function::current().arguments()) {
        if (v->is_uniform_argument() && v->uniform_data() == data_ref) { return v.get(); }
    }
    auto v = std::make_unique<Variable>();
    v->_type = type;
    v->_tag = VariableTag::UNIFORM;
    v->_uniform_data = data_ref;
    v->_uid = Function::current().next_uid();
    return Function::current().add_argument(std::move(v));
}

const Variable *Variable::make_immutable_argument(const TypeDesc *type, const std::vector<std::byte> &data) noexcept {
    for (auto &&v : Function::current().arguments()) {
        if (v->is_immutable_argument() &&
            v->immutable_data().size() == data.size() &&
            std::memcmp(v->immutable_data().data(), data.data(), data.size()) == 0) { return v.get(); }
    }
    auto v = std::make_unique<Variable>();
    v->_type = type;
    v->_tag = VariableTag::IMMUTABLE;
    v->_immutable_data = data;
    v->_uid = Function::current().next_uid();
    return Function::current().add_argument(std::move(v));
}

const Variable *Variable::make_temporary(const TypeDesc *type, std::unique_ptr<Expression> expression) noexcept {
    auto v = std::make_unique<Variable>();
    v->_type = type;
    v->_tag = VariableTag::TEMPORARY;
    v->_expression = std::move(expression);
    v->_uid = Function::current().next_uid();
    return Function::current().add_variable(std::move(v));
}

const Variable *Variable::make_builtin(VariableTag tag) noexcept {
    assert(tag == VariableTag::THREAD_ID || tag == VariableTag::THREAD_XY);
    for (auto &&v : Function::current().builtins()) {
        if (v->tag() == tag) { return v.get(); }
    }
    auto v = std::make_unique<Variable>();
    v->_tag = tag;
    v->_uid = Function::current().next_uid();
    if (tag == VariableTag::THREAD_ID) { v->_type = type_desc<uint>; }
    else if (tag == VariableTag::THREAD_XY) { v->_type = type_desc<uint2>; }
    return Function::current().add_builtin(std::move(v));
}

const Variable *Variable::make_buffer_argument(const TypeDesc *type, const std::shared_ptr<Buffer> &buffer) noexcept {
    for (auto &&v : Function::current().arguments()) {
        if (v->is_buffer_argument() && v->buffer() == buffer.get()) { return v.get(); }
    }
    auto v = std::make_unique<Variable>();
    v->_type = type;
    v->_tag = VariableTag::BUFFER;
    v->_buffer = buffer;
    v->_uid = Function::current().next_uid();
    return Function::current().add_argument(std::move(v));
}

const Variable *Variable::make_texture_argument(const std::shared_ptr<Texture> &texture) noexcept {
    for (auto &&v : Function::current().arguments()) {
        if (v->is_texture_argument() && v->texture() == texture.get()) { return v.get(); }
    }
    auto v = std::make_unique<Variable>();
    v->_tag = VariableTag::TEXTURE;
    v->_texture = texture;
    v->_uid = Function::current().next_uid();
    return Function::current().add_argument(std::move(v));
}

}

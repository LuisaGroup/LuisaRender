//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <variant>

#include <compute/type_desc.h>
#include <compute/buffer.h>
#include <compute/texture.h>

namespace luisa::compute::dsl {

class Expression;
class Function;

namespace detail {

template<typename T>
using EnableIfLiteralOperand = std::enable_if_t<
    std::is_convertible_v<T, std::variant<
        bool, float, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>>, int>;
    
}

enum struct VariableTag {
    
    INVALID,    // invalid
    
    // for arguments
    BUFFER,     // device buffers
    TEXTURE,    // textures
    UNIFORM,    // uniforms
    IMMUTABLE,  // immutable data, i.e. constant uniforms
    
    // for local variables
    LOCAL,      // local variables
    
    // for expression nodes
    TEMPORARY,  // temporary variables, i.e. expression nodes
    
    // for builtin variables
    THREAD_ID,  // built-in thread id
    THREAD_XY,  // built-in thread coord
    THREAD_XYZ  // built-in thread 3d coord
};

class Variable {

protected:
    const TypeDesc *_type{nullptr};
    uint32_t _uid{0u};
    VariableTag _tag{VariableTag::INVALID};
    
    // For kernel argument bindings
    BufferView<std::byte> _buffer;
    Texture *_texture{nullptr};
    void *_data_ref{nullptr};
    std::vector<std::byte> _immutable_data;
    
    // For temporary variables in expressions
    Expression *_expression{nullptr};

public:
    // Empty (i.e. invalid) variables, do not use unless necessary
    Variable() noexcept = default;
    
    // Local variables
    Variable(const TypeDesc *type, uint32_t uid) noexcept: _type{type}, _uid{uid}, _tag{VariableTag::LOCAL} {
        LUISA_ERROR_IF(is_ptr_or_ref(_type), "Declaring local variable v", uid, " as a pointer or reference is not allowed.");
    }
    
    // Buffer arguments
    Variable(const TypeDesc *type, uint32_t uid, Buffer *buffer, size_t offset, size_t size) noexcept
        : _type{type}, _uid{uid}, _buffer{buffer->view<std::byte>(offset, size)}, _tag{VariableTag::BUFFER} {
        
        LUISA_ERROR_IF_NOT(is_ptr_or_ref(_type), "Argument v", uid, " bound to a buffer is not declared as a pointer or reference.");
    }
    
    // Texture arguments
    Variable(const TypeDesc *type, uint32_t uid, Texture *texture) noexcept: _type{type}, _uid{uid}, _texture{texture}, _tag{VariableTag::TEXTURE} {
        LUISA_ERROR_IF_NOT(_type->type == TypeCatalog::TEXTURE, "Argument v", uid, " bound to a texture is not declared as a texture.");
    }
    
    // Immutable uniforms
    Variable(const TypeDesc *type, uint32_t uid, const void *data, size_t size) noexcept: _type{type}, _uid{uid}, _tag{VariableTag::IMMUTABLE} {
        LUISA_ERROR_IF(is_ptr_or_ref(_type) || _type->type == TypeCatalog::TEXTURE, "Argument v", uid, " bound to constant data is not declared as is.");
        _immutable_data.resize(size);
        std::memmove(_immutable_data.data(), data, size);
    }
    
    // Uniforms
    Variable(const TypeDesc *type, uint32_t uid, void *data_ref) noexcept : _type{type}, _uid{uid}, _data_ref{data_ref}, _tag{VariableTag::UNIFORM} {
        LUISA_ERROR_IF(is_ptr_or_ref(_type) || _type->type == TypeCatalog::TEXTURE, "Argument v", uid, " bound to constant data is not declared as is.");
    }
    
    // Built-in variables
    Variable(const TypeDesc *type, VariableTag tag) noexcept
        : _type{type}, _tag{tag} {}
    
    // For temporary variables, i.e. expression nodes
    explicit Variable(Expression *expr) noexcept
        : _expression{expr}, _tag{VariableTag::TEMPORARY} {}
    
    Variable(Variable &&) = default;
    Variable(const Variable &) = default;
    
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    
    [[nodiscard]] bool is_valid() const noexcept { return _tag != VariableTag::INVALID; }
    [[nodiscard]] bool is_temporary() const noexcept { return _tag == VariableTag::TEMPORARY; }
    [[nodiscard]] bool is_local() const noexcept { return _tag == VariableTag::LOCAL; }
    
    [[nodiscard]] bool is_buffer_argument() const noexcept { return _tag == VariableTag::BUFFER; }
    [[nodiscard]] bool is_texture_argument() const noexcept { return _tag == VariableTag::TEXTURE; }
    [[nodiscard]] bool is_uniform_argument() const noexcept { return _tag == VariableTag::UNIFORM; }
    [[nodiscard]] bool is_immutable_argument() const noexcept { return _tag == VariableTag::IMMUTABLE; }
    
    [[nodiscard]] bool is_argument() const noexcept {
        return is_buffer_argument() || is_texture_argument() || is_uniform_argument() || is_immutable_argument();
    }
    
    [[nodiscard]] bool is_thread_id() const noexcept { return _tag == VariableTag::THREAD_ID; }
    [[nodiscard]] bool is_thread_xy() const noexcept { return _tag == VariableTag::THREAD_XY; }
    [[nodiscard]] bool is_thread_xyz() const noexcept { return _tag == VariableTag::THREAD_XYZ; }
    [[nodiscard]] bool is_builtin() const noexcept { return is_thread_id(); }
    
    [[nodiscard]] Expression *expression() const noexcept { return _expression; }
    [[nodiscard]] const TypeDesc *type() const noexcept { return _type; }
    [[nodiscard]] uint32_t uid() const noexcept { return _uid; }
    
    // for buffers
    [[nodiscard]] BufferView<std::byte> buffer() const noexcept { return _buffer; }
    
    // for textures
    [[nodiscard]] Texture *texture() const noexcept { return _texture; }
    
    // for uniforms
    [[nodiscard]] void *uniform_data() const noexcept { return _data_ref; }
    
    // for immutable data
    [[nodiscard]] const std::vector<std::byte> &immutable_data() const noexcept { return _immutable_data; }
    
    // Member access operators
    [[nodiscard]] Variable member(std::string m) const noexcept;
    [[nodiscard]] Variable operator[](std::string m) const noexcept { return member(std::move(m)); }
    
    // Convenient methods for accessing vector members
    [[nodiscard]] Variable x() const noexcept { return member("x"); }
    [[nodiscard]] Variable y() const noexcept { return member("y"); }
    [[nodiscard]] Variable z() const noexcept { return member("z"); }
    [[nodiscard]] Variable w() const noexcept { return member("w"); }
    [[nodiscard]] Variable r() const noexcept { return member("x"); }
    [[nodiscard]] Variable g() const noexcept { return member("y"); }
    [[nodiscard]] Variable b() const noexcept { return member("z"); }
    [[nodiscard]] Variable a() const noexcept { return member("w"); }
    
    [[nodiscard]] Variable operator+() const noexcept;
    [[nodiscard]] Variable operator-() const noexcept;
    [[nodiscard]] Variable operator~() const noexcept;
    [[nodiscard]] Variable operator!() const noexcept;
    [[nodiscard]] Variable operator*() const noexcept;
    [[nodiscard]] Variable operator&() const noexcept;

#define MAKE_BINARY_OPERATOR_DECL(op)                                                                                    \
[[nodiscard]] Variable operator op(Variable rhs) const noexcept;                                                         \
template<typename T, detail::EnableIfLiteralOperand<T> = 0> [[nodiscard]] Variable operator op(T &&rhs) const noexcept;  \

#define MAKE_ASSIGNMENT_OPERATOR_DECL(op)                                                                                \
void operator op(Variable rhs) const noexcept;                                                                           \
template<typename T, detail::EnableIfLiteralOperand<T> = 0> void operator op(T &&rhs) const noexcept;                    \

    LUISA_MAP_MACRO(MAKE_BINARY_OPERATOR_DECL, +, -, *, /, %, <<, >>, &, |, ^, &&, ||, ==, !=, <,>, <=, >=, [])
    LUISA_MAP_MACRO(MAKE_ASSIGNMENT_OPERATOR_DECL, =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=)

#undef MAKE_BINARY_OPERATOR_DECL
#undef MAKE_ASSIGNMENT_OPERATOR_DECL

};

#define $(m) member(#m)

}

//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <variant>
#include <compute/type_desc.h>

namespace luisa::compute::dsl {

class Buffer;
class Texture;
class Expression;

enum struct ResourceUsage : uint32_t {
    NONE,
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE,
    SAMPLE,  // For textures only
};

// new version of dsl
enum struct VariableTag {
    
    EMPTY,// should not appear
    
    // for arguments
    BUFFER,   // device buffers
    TEXTURE,  // textures
    UNIFORM,  // uniforms
    IMMUTABLE,// immutable data, i.e. constant uniforms
    
    // for variable declarations
    LOCAL,      // local variables
    THREADGROUP,// threadgroup variables
    
    // for expression nodes
    TEMPORARY,// temporary variables, i.e. expression nodes
    
    // for builtin variables
    THREAD_ID,// built-in thread id
    THREAD_XY,// built-in thread coord
};

class Variable {

public:
    static constexpr auto resource_read_bit = 1u;
    static constexpr auto resource_write_bit = 2u;
    static constexpr auto resource_sample_bit = 4u;

private:
    const TypeDesc *_type{nullptr};
    uint32_t _uid{0u};
    VariableTag _tag{VariableTag::EMPTY};
    
    // For kernel argument bindings
    std::shared_ptr<Buffer> _buffer{nullptr};
    std::shared_ptr<Texture> _texture{nullptr};
    std::vector<std::byte> _immutable_data;
    const void *_uniform_data{nullptr};
    
    // argument usage
    mutable uint32_t _usage{0u};
    
    // For temporary variables in expressions
    std::unique_ptr<Expression> _expression{nullptr};

public:
    [[nodiscard]] static const Variable *make_builtin(VariableTag tag) noexcept;
    [[nodiscard]] static const Variable *make_local_variable(const TypeDesc *type) noexcept;
    [[nodiscard]] static const Variable *make_threadgroup_variable(const TypeDesc *type) noexcept;
    [[nodiscard]] static const Variable *make_buffer_argument(const TypeDesc *type, const std::shared_ptr<Buffer> &buffer) noexcept;
    [[nodiscard]] static const Variable *make_texture_argument(const std::shared_ptr<Texture> &texture) noexcept;
    [[nodiscard]] static const Variable *make_uniform_argument(const TypeDesc *type, const void *data_ref) noexcept;
    [[nodiscard]] static const Variable *make_immutable_argument(const TypeDesc *type, const std::vector<std::byte> &data) noexcept;
    [[nodiscard]] static const Variable *make_temporary(const TypeDesc *type, std::unique_ptr<Expression> expression) noexcept;
    
    [[nodiscard]] const TypeDesc *type() const noexcept { return _type; }
    [[nodiscard]] uint uid() const noexcept { return _uid; }
    [[nodiscard]] VariableTag tag() const noexcept { return _tag; }
    
    [[nodiscard]] bool is_temporary() const noexcept { return _tag == VariableTag::TEMPORARY; }
    [[nodiscard]] bool is_local() const noexcept { return _tag == VariableTag::LOCAL; }
    [[nodiscard]] bool is_threadgroup() const noexcept { return _tag == VariableTag::THREADGROUP; }
    
    [[nodiscard]] bool is_buffer_argument() const noexcept { return _tag == VariableTag::BUFFER; }
    [[nodiscard]] bool is_texture_argument() const noexcept { return _tag == VariableTag::TEXTURE; }
    [[nodiscard]] bool is_uniform_argument() const noexcept { return _tag == VariableTag::UNIFORM; }
    [[nodiscard]] bool is_immutable_argument() const noexcept { return _tag == VariableTag::IMMUTABLE; }
    
    [[nodiscard]] bool is_argument() const noexcept {
        return is_buffer_argument() || is_texture_argument() || is_uniform_argument() || is_immutable_argument();
    }
    
    [[nodiscard]] bool is_thread_id() const noexcept { return _tag == VariableTag::THREAD_ID; }
    [[nodiscard]] bool is_thread_xy() const noexcept { return _tag == VariableTag::THREAD_XY; }
    [[nodiscard]] bool is_builtin() const noexcept { return is_thread_id() || is_thread_xy(); }
    
    [[nodiscard]] Expression *expression() const noexcept { return _expression.get(); }
    
    [[nodiscard]] Texture *texture() const noexcept { return _texture.get(); }
    [[nodiscard]] Buffer *buffer() const noexcept { return _buffer.get(); }
    [[nodiscard]] const std::vector<std::byte> &immutable_data() const noexcept { return _immutable_data; }
    [[nodiscard]] const void *uniform_data() const noexcept { return _uniform_data; }
    
    void mark_read() const noexcept { _usage |= resource_read_bit; }
    void mark_write() const noexcept { _usage |= resource_write_bit; }
    void mark_sample() const noexcept { _usage |= resource_sample_bit; }
    
    [[nodiscard]] ResourceUsage usage() const noexcept {
        
        bool read = _usage & resource_read_bit;
        bool write = _usage & resource_write_bit;
        bool sample = _usage & resource_sample_bit;
        
        assert(!(sample && read) && !(sample && write));
        assert(!sample || is_texture_argument());
        
        if (read && write) { return ResourceUsage::READ_WRITE; }
        if (read) { return ResourceUsage::READ_ONLY; }
        if (write) { return ResourceUsage::WRITE_ONLY; }
        if (sample) { return ResourceUsage::SAMPLE; }
        return ResourceUsage::NONE;
    }
};

}// namespace luisa::compute::dsl

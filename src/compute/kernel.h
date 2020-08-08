//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <variant>
#include <vector>
#include <string_view>

#include <core/concepts.h>

#include <core/data_types.h>
#include <compute/buffer.h>

namespace luisa {

enum struct KernelArgumentTag {
    BUFFER, UNIFORM
};

class KernelArgument {

private:
    KernelArgumentTag _tag;
    
    // buffer data
    TypelessBuffer *_buffer{nullptr};
    size_t _buffer_offset{0u};
    
    // uniform data
    std::vector<std::byte> _data;

public:
    template<typename T>
    KernelArgument(BufferView<T> view) noexcept
        : _tag{KernelArgumentTag::BUFFER}, _buffer{std::addressof(view.typeless_buffer())}, _buffer_offset{view.byte_offset()} {}
    
    template<typename T>
    KernelArgument(T &&data) noexcept : _tag{KernelArgumentTag::UNIFORM} {
        _data.resize(sizeof(T));
        std::memmove(_data.data(), &data, _data.size());
    }
    
    [[nodiscard]] KernelArgumentTag tag() const noexcept { return _tag; }
    [[nodiscard]] TypelessBuffer *buffer() const noexcept { return _buffer; }
    [[nodiscard]] size_t buffer_offset() const noexcept { return _buffer_offset; }
    [[nodiscard]] const std::vector<std::byte> &data() const noexcept { return _data; }
};

struct KernelArgumentEncoder : Noncopyable {
    
    virtual ~KernelArgumentEncoder() noexcept = default;
    
    virtual void set_buffer(std::string_view argument_name, TypelessBuffer &buffer, size_t offset) = 0;
    virtual void set_bytes(std::string_view argument_name, const void *data, size_t size) = 0;
    
    void operator()(std::string_view argument_name, TypelessBuffer &buffer, size_t offset = 0ul) {
        set_buffer(argument_name, buffer, offset);
    }
    
    template<typename T>
    void operator()(std::string_view argument_name, BufferView<T> buffer_view) {
        set_buffer(argument_name, buffer_view.typeless_buffer(), buffer_view.byte_offset());
    }
    
    template<typename T>
    void operator()(std::string_view argument_name, Buffer<T> &buffer) {
        set_buffer(argument_name, buffer.view().typeless_buffer(), buffer.view().byte_offset());
    }
    
    void operator()(std::string_view argument_name, const void *bytes, size_t size) {
        set_bytes(argument_name, bytes, size);
    }
    
    template<typename T>
    void operator()(std::string_view argument_name, T data) {
        set_bytes(argument_name, &data, sizeof(T));
    }
    
    template<typename T>
    void operator()(std::string_view argument_name, const T *data, size_t count) {
        set_bytes(argument_name, data, sizeof(T) * count);
    }
};

struct Kernel : Noncopyable {
    virtual ~Kernel() = default;
};

struct KernelDispatcher : Noncopyable {
    
    virtual ~KernelDispatcher() noexcept = default;
    
//    virtual void operator()(Kernel &kernel, uint2 threadgroups, uint2 threadgroup_size, const std::vector<KernelArgument> &arguments) = 0;
    
    virtual void operator()(Kernel &kernel, uint2 threadgroups, uint2 threadgroup_size, std::function<void(KernelArgumentEncoder &)> encode) = 0;
    virtual void operator()(Kernel &kernel, uint threadgroups, uint threadgroup_size, std::function<void(KernelArgumentEncoder &)> encode) {
        (*this)(kernel, {threadgroups, 1u}, {threadgroup_size, 1u}, std::move(encode));
    }
    
    void operator()(Kernel &kernel, uint2 extent, std::function<void(KernelArgumentEncoder &)> encode) {
        auto threadgroup_size = make_uint2(16u, 16u);
        auto threadgroups = (extent + threadgroup_size - make_uint2(1u)) / threadgroup_size;
        (*this)(kernel, threadgroups, threadgroup_size, std::move(encode));
    }
    
    void operator()(Kernel &kernel, uint extent, std::function<void(KernelArgumentEncoder &)> encode) {
        auto threadgroup_size = 128u;
        auto threadgroups = (extent + threadgroup_size - 1u) / threadgroup_size;
        (*this)(kernel, threadgroups, threadgroup_size, std::move(encode));
    }
};

}

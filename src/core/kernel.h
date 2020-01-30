//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <string_view>
#include <util/noncopyable.h>
#include "buffer.h"
#include "texture.h"

namespace luisa {

struct KernelArgumentProxy : Noncopyable {
    virtual ~KernelArgumentProxy() = default;
    virtual void set_buffer(Buffer &buffer) { set_buffer(buffer, 0ul); }
    virtual void set_buffer(Buffer &buffer, size_t offset) = 0;
    virtual void set_texture(Texture &texture) = 0;
    virtual void set_bytes(const void *bytes, size_t size) = 0;
};

struct KernelArgumentBufferMemberProxy : Noncopyable {
    virtual ~KernelArgumentBufferMemberProxy() = default;
    virtual void set_buffer(Buffer &argument_buffer, size_t argument_buffer_offset, Buffer &buffer) { set_buffer(argument_buffer, argument_buffer_offset, buffer, 0ul); }
    virtual void set_buffer(Buffer &argument_buffer, size_t argument_buffer_offset, Buffer &buffer, size_t offset) = 0;
    virtual void set_texture(Buffer &argument_buffer, size_t argument_buffer_offset, Texture &texture) = 0;
    virtual void set_bytes(Buffer &argument_buffer, size_t argument_buffer_offset, const void *bytes, size_t size) = 0;
};

struct KernelArgumentEncoder : Noncopyable {
    virtual ~KernelArgumentEncoder() noexcept = default;
    [[nodiscard]] virtual std::unique_ptr<KernelArgumentProxy> operator[](std::string_view argument_name) = 0;
};

struct KernelArgumentBufferEncoder : Noncopyable {
    virtual ~KernelArgumentBufferEncoder() noexcept = default;
    [[nodiscard]] virtual std::unique_ptr<KernelArgumentBufferMemberProxy> operator[](std::string_view member_name) = 0;
    [[nodiscard]] virtual size_t element_size() const = 0;
    [[nodiscard]] virtual size_t element_alignment() const = 0;
    [[nodiscard]] virtual size_t aligned_element_size() const {
        auto size = element_size();
        auto alignment = element_alignment();
        return (size + alignment - 1ul) / alignment * alignment;
    }
};

struct Kernel : Noncopyable {
    virtual ~Kernel() = default;
    [[nodiscard]] virtual std::unique_ptr<KernelArgumentBufferEncoder> argument_buffer_encoder(std::string_view argument_name) = 0;
};

struct KernelDispatcher : Noncopyable {
    virtual ~KernelDispatcher() noexcept = default;
    virtual void operator()(Kernel &kernel, uint2 threadgroups, uint2 threadgroup_size, std::function<void(KernelArgumentEncoder &)> encode) = 0;
    virtual void operator()(Kernel &kernel, uint threadgroups, uint threadgroup_size, std::function<void(KernelArgumentEncoder &)> encode) {
        (*this)(kernel, {threadgroups, 1u}, {threadgroup_size, 1u}, std::move(encode));
    }
};

}
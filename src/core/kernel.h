//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <string_view>
#include <util/noncopyable.h>
#include <compatibility.h>
#include "buffer.h"
#include "texture.h"

struct KernelArgumentProxy : Noncopyable {
    virtual ~KernelArgumentProxy() = default;
    virtual void set_buffer(Buffer &buffer) = 0;
    virtual void set_texture(Texture &texture) = 0;
    virtual void set_bytes(const void *bytes, size_t size) = 0;
};

class KernelArgumentProxyWrapper {

private:
    std::unique_ptr<KernelArgumentProxy> _proxy;

public:
    explicit KernelArgumentProxyWrapper(std::unique_ptr<KernelArgumentProxy> proxy) noexcept : _proxy{std::move(proxy)} {}
    
    template<typename T>
    KernelArgumentProxyWrapper &operator=(T &&value) {
        if constexpr (std::is_base_of_v<Buffer, T>) {
            _proxy->set_buffer(std::forward<T>(value));
        } else if constexpr (std::is_base_of_v<Texture, T>) {
            _proxy->set_texture(std::forward<T>(value));
        } else {
            _proxy->set_bytes(&value, sizeof(value));
        }
        return *this;
    }
};

struct KernelArgumentEncoder : Noncopyable {
    virtual ~KernelArgumentEncoder() noexcept = default;
    [[nodiscard]] virtual KernelArgumentProxyWrapper operator[](std::string_view argument_name) = 0;
};

struct Kernel : Noncopyable {
    virtual ~Kernel() = default;
};

struct KernelDispatcher : Noncopyable {
    virtual ~KernelDispatcher() noexcept = default;
    virtual void operator()(Kernel &kernel, uint2 grids, uint2 grid_size, std::function<void(KernelArgumentEncoder &)> encode) = 0;
};

//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <limits>
#include <functional>

#include <core/concepts.h>
#include <compute/dispatcher.h>

namespace luisa::compute {

template<typename T>
class BufferView;

class Buffer : Noncopyable {

public:
    static constexpr auto npos = std::numeric_limits<size_t>::max();

protected:
    size_t _size;

public:
    explicit Buffer(size_t size) noexcept: _size{size} {}
    virtual ~Buffer() noexcept = default;
    
    [[nodiscard]] size_t size() const noexcept { return _size; }
    
    template<typename T>
    [[nodiscard]] auto view(size_t offset = 0u, size_t size = npos) noexcept;
    
    virtual void upload(Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) = 0;
    virtual void download(Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) const = 0;
};

template<typename T>
class BufferView {

public:
    static constexpr auto npos = std::numeric_limits<size_t>::max();

private:
    Buffer *_buffer{nullptr};
    size_t _offset{0u};
    size_t _size{0u};

public:
    BufferView() noexcept = default;
    
    explicit BufferView(Buffer *buffer, size_t offset = 0u, size_t size = npos) noexcept: _buffer{buffer}, _offset{offset}, _size{size} {
        if (size == npos) { _size = (_buffer->size() - byte_offset()) % sizeof(T); }
    }
    
    [[nodiscard]] BufferView subview(size_t offset, size_t size = npos) noexcept { return {_buffer, _offset + offset, size}; }
    
    [[nodiscard]] bool empty() const noexcept { return _buffer == nullptr || _size == 0u; }
    [[nodiscard]] Buffer *buffer() const noexcept { return _buffer; }
    [[nodiscard]] size_t offset() const noexcept { return _offset; }
    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] size_t byte_offset() const noexcept { return _offset * sizeof(T); }
    [[nodiscard]] size_t byte_size() const noexcept { return _size * sizeof(T); }
    
    void upload(Dispatcher &dispatcher, const void *host_data) { _buffer->upload(dispatcher, byte_offset(), byte_size(), host_data); }
    void download(Dispatcher &dispatcher, void *host_buffer) const { _buffer->download(dispatcher, byte_offset(), byte_size(), host_buffer); }
};

template<typename T>
inline auto Buffer::view(size_t offset, size_t size) noexcept {
    return BufferView<T>{this, offset, size};
}

}

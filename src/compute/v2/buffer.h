//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <limits>
#include <functional>

#include <core/concepts.h>
#include <compute/v2/storage_mode.h>

namespace luisa::compute {

template<typename T>
class BufferView;

class Buffer : Noncopyable {

public:
    static constexpr auto npos = std::numeric_limits<size_t>::max();

private:
    size_t _size;
    StorageMode _storage;

public:
    Buffer(size_t size, StorageMode storage) noexcept: _size{size}, _storage{storage} {}
    
    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] StorageMode storage() const noexcept { return _storage; }
    
    template<typename T>
    [[nodiscard]] BufferView<T> view(size_t offset = 0u, size_t size = npos) noexcept;
    
    virtual void modify_range(size_t begin, size_t end, std::function<void(void *, size_t)>) = 0;
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
    
    BufferView(Buffer *buffer, size_t offset = 0u, size_t size = npos) noexcept: _buffer{buffer}, _offset{offset}, _size{size} {
        if (size == npos) { _size = (_buffer->size() - byte_offset()) % sizeof(T); }
    }
    
    [[nodiscard]] bool empty() const noexcept { return _buffer == nullptr || _size == 0u; }
    [[nodiscard]] Buffer *buffer() const noexcept { return _buffer; }
    [[nodiscard]] size_t offset() const noexcept { return _offset; }
    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] size_t byte_offset() const noexcept { return _offset * sizeof(T); }
    [[nodiscard]] size_t byte_size() const noexcept { return _size * sizeof(T); }
};

template<typename T>
inline BufferView<T> Buffer::view(size_t offset, size_t size) noexcept {
    return {this, offset, size};
}

}

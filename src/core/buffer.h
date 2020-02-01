//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <vector>
#include <util/noncopyable.h>

namespace luisa {

enum struct BufferStorage {
    DEVICE_PRIVATE,
    MANAGED
};

class Buffer;

template<typename T>
class BufferView {

private:
    Buffer *_buffer;
    size_t _element_offset;
    size_t _element_count;

public:
    BufferView(Buffer *buffer, size_t element_offset, size_t element_count)
        : _buffer{buffer}, _element_offset{element_offset}, _element_count{element_count} {}
    
    [[nodiscard]] size_t size() const noexcept { return _element_count * sizeof(T); }
    [[nodiscard]] size_t offset() const noexcept { return _element_offset * sizeof(T); }
    [[nodiscard]] size_t element_count() const noexcept { return _element_count; }
    [[nodiscard]] size_t element_offset() const noexcept { return _element_offset; }
    [[nodiscard]] Buffer &buffer() noexcept { return *_buffer; }
};

class Buffer : Noncopyable {

protected:
    size_t _capacity;
    BufferStorage _storage;

public:
    Buffer(size_t capacity, BufferStorage storage) noexcept : _capacity{capacity}, _storage{storage} {};
    virtual ~Buffer() noexcept = default;
    virtual void upload(const void *host_data, size_t size, size_t offset) = 0;
    virtual void upload(const void *host_data, size_t size) { upload(host_data, size, 0ul); }
    virtual void upload() { upload(nullptr, 0ul); }
    virtual void synchronize(struct KernelDispatcher &dispatch) = 0;
    [[nodiscard]] virtual const void *data() const { return const_cast<Buffer *>(this)->data(); }
    [[nodiscard]] virtual void *data() = 0;
    [[nodiscard]] virtual size_t capacity() const noexcept { return _capacity; }
    
    template<typename T>
    [[nodiscard]] BufferView<T> view(size_t element_offset = 0ul) noexcept {
        assert(_capacity % sizeof(T) == 0ul);
        return {this, element_offset, _capacity / sizeof(T) - element_offset};
    };
    
    template<typename T>
    [[nodiscard]] BufferView<T> view(size_t element_offset, size_t element_count) noexcept {
        assert((element_count + element_offset) * sizeof(T) <= _capacity);
        return {this, element_offset, element_count};
    }
};

}

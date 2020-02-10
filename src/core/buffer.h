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
    
    [[nodiscard]] size_t byte_size() const noexcept { return _element_count * sizeof(T); }
    [[nodiscard]] size_t byte_offset() const noexcept { return _element_offset * sizeof(T); }
    [[nodiscard]] size_t element_count() const noexcept { return _element_count; }
    [[nodiscard]] size_t element_offset() const noexcept { return _element_offset; }
    [[nodiscard]] Buffer &buffer() noexcept { return *_buffer; }
    [[nodiscard]] T *data();
    [[nodiscard]] const T *data() const { return const_cast<BufferView<T> *>(this)->data(); }
    [[nodiscard]] T &operator[](size_t index) { return data()[index]; }
    [[nodiscard]] const T &operator[](size_t index) const { return data()[index]; }
    void upload();
};

class Buffer : Noncopyable {

protected:
    size_t _capacity;
    BufferStorage _storage;

public:
    Buffer(size_t capacity, BufferStorage storage) noexcept : _capacity{capacity}, _storage{storage} {};
    virtual ~Buffer() noexcept = default;
    virtual void upload(size_t offset, size_t size) = 0;
    virtual void upload() { upload(0ul, capacity()); }
    virtual void synchronize(struct KernelDispatcher &dispatch) = 0;
    [[nodiscard]] virtual const void *data() const { return const_cast<Buffer *>(this)->data(); }
    [[nodiscard]] virtual void *data() = 0;
    [[nodiscard]] size_t capacity() const noexcept { return _capacity; }
    [[nodiscard]] BufferStorage storage() const noexcept { return _storage; }
    
    template<typename T>
    [[nodiscard]] auto view(size_t element_offset = 0ul) noexcept {
        assert(_capacity % sizeof(T) == 0ul);
        return BufferView<T>{this, element_offset, _capacity / sizeof(T) - element_offset};
    };
    
    template<typename T>
    [[nodiscard]] BufferView<T> view(size_t element_offset, size_t element_count) noexcept {
        assert((element_count + element_offset) * sizeof(T) <= _capacity);
        return BufferView<T>{this, element_offset, element_count};
    }
};

template<typename T>
T *BufferView<T>::data() {
    return &reinterpret_cast<T *>(_buffer->data())[_element_offset];
}

template<typename T>
void BufferView<T>::upload() { _buffer->upload(byte_offset(), byte_size()); }

}

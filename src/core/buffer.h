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

class TypelessBuffer;

template<typename T>
class BufferView {

private:
    TypelessBuffer *_buffer;
    size_t _element_offset;
    size_t _element_count;

public:
    BufferView(TypelessBuffer *buffer, size_t element_offset, size_t element_count)
        : _buffer{buffer}, _element_offset{element_offset}, _element_count{element_count} {}
    
    [[nodiscard]] size_t byte_size() const noexcept { return _element_count * sizeof(T); }
    [[nodiscard]] size_t byte_offset() const noexcept { return _element_offset * sizeof(T); }
    [[nodiscard]] size_t element_count() const noexcept { return _element_count; }
    [[nodiscard]] size_t element_offset() const noexcept { return _element_offset; }
    [[nodiscard]] TypelessBuffer &typeless_buffer() noexcept { return *_buffer; }
    [[nodiscard]] T *data();
    [[nodiscard]] const T *data() const { return const_cast<BufferView<T> *>(this)->data(); }
    [[nodiscard]] T &operator[](size_t index) { return data()[index]; }
    [[nodiscard]] const T &operator[](size_t index) const { return data()[index]; }
    void upload();
};

class TypelessBuffer : Noncopyable {

protected:
    size_t _capacity;
    BufferStorage _storage;

public:
    TypelessBuffer(size_t capacity, BufferStorage storage) noexcept : _capacity{capacity}, _storage{storage} {};
    virtual ~TypelessBuffer() noexcept = default;
    virtual void upload(size_t offset, size_t size) = 0;
    virtual void upload() { upload(0ul, capacity()); }
    virtual void synchronize(struct KernelDispatcher &dispatch) = 0;
    [[nodiscard]] virtual const void *data() const { return const_cast<TypelessBuffer *>(this)->data(); }
    [[nodiscard]] virtual void *data() = 0;
    [[nodiscard]] size_t capacity() const noexcept { return _capacity; }
    [[nodiscard]] BufferStorage storage() const noexcept { return _storage; }
};

template<typename Element>
class Buffer {

private:
    std::unique_ptr<TypelessBuffer> _typeless_buffer;

public:
    Buffer() = delete;
    explicit Buffer(std::unique_ptr<TypelessBuffer> mem) noexcept : _typeless_buffer{std::move(mem)} {
        assert(_typeless_buffer->capacity() % sizeof(Element) == 0);
    }
    
    [[nodiscard]] auto view(size_t element_offset, size_t element_count) noexcept {
        assert((element_count + element_offset) * sizeof(Element) <= _typeless_buffer->capacity());
        return BufferView<Element>{_typeless_buffer.get(), element_offset, element_count};
    }
    [[nodiscard]] auto view(size_t element_offset = 0ul) noexcept {
        assert(_capacity % sizeof(Element) == 0ul);
        return BufferView<Element>{_typeless_buffer.get(), element_offset, _typeless_buffer->capacity() / sizeof(Element) - element_offset};
    };
    
    template<typename T>
    [[nodiscard]] auto view_as(size_t element_offset = 0ul) noexcept {
        assert(_capacity % sizeof(T) == 0ul);
        return BufferView<T>{_typeless_buffer.get(), element_offset, _typeless_buffer->capacity() / sizeof(T) - element_offset};
    };
    
    template<typename T>
    [[nodiscard]] BufferView<T> view_as(size_t element_offset, size_t element_count) noexcept {
        assert((element_count + element_offset) * sizeof(T) <= _typeless_buffer->capacity());
        return BufferView<T>{_typeless_buffer.get(), element_offset, element_count};
    }
    
    void synchronize(struct KernelDispatcher &dispatch) { _typeless_buffer->synchronize(dispatch); }
    void upload() { _typeless_buffer->upload(); }
    void upload(size_t offset, size_t size) { view(offset, size).upload(); }
    
    [[nodiscard]] size_t capacity() const noexcept { return _typeless_buffer->capacity() / sizeof(Element); }
};

template<typename T>
T *BufferView<T>::data() {
    return &reinterpret_cast<T *>(_buffer->data())[_element_offset];
}

template<typename T>
void BufferView<T>::upload() { _buffer->upload(byte_offset(), byte_size()); }

}

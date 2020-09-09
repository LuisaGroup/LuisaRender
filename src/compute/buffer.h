//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <iostream>
#include <limits>
#include <functional>
#include <utility>

#include <core/concepts.h>
#include <compute/dispatcher.h>
#include <compute/dsl.h>

namespace luisa::compute {

template<typename T>
class BufferView;

class Buffer : private Noncopyable, public std::enable_shared_from_this<Buffer> {

public:
    static constexpr auto npos = std::numeric_limits<size_t>::max();

protected:
    size_t _size;

public:
    explicit Buffer(size_t size) noexcept
        : _size{size} {}
    virtual ~Buffer() noexcept = default;
    
    [[nodiscard]] size_t size() const noexcept { return _size; }
    
    template<typename T>
    [[nodiscard]] auto view(size_t offset = 0u, size_t size = npos) noexcept;
    
    virtual void upload(Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) = 0;
    virtual void download(Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) = 0;
    virtual void clear_cache() noexcept = 0;
    
    virtual void with_cache(compute::Dispatcher &dispatch, const std::function<void(void *)> &modify, size_t offset, size_t length) = 0;
};

template<typename T>
class BufferView {

public:
    static constexpr auto npos = std::numeric_limits<size_t>::max();

private:
    std::shared_ptr<Buffer> _buffer;
    size_t _offset{0u};
    size_t _size{0u};
    
    void _copy_from(Dispatcher &dispatcher, const void *host_data) const { _buffer->upload(dispatcher, byte_offset(), byte_size(), host_data); }
    void _copy_to(Dispatcher &dispatcher, void *host_buffer) const { _buffer->download(dispatcher, byte_offset(), byte_size(), host_buffer); }

public:
    BufferView() noexcept = default;
    
    explicit BufferView(std::shared_ptr<Buffer> buffer, size_t offset = 0u, size_t size = npos) noexcept: _buffer{std::move(buffer)}, _offset{offset}, _size{size} {
        if (_size == npos) { _size = (_buffer->size() - byte_offset()) / sizeof(T); }
    }
    
    [[nodiscard]] BufferView subview(size_t offset, size_t size = npos) const noexcept { return {_buffer, _offset + offset, size}; }
    
    [[nodiscard]] bool empty() const noexcept { return _buffer == nullptr || _size == 0u; }
    [[nodiscard]] Buffer *buffer() const noexcept { return _buffer.get(); }
    [[nodiscard]] size_t offset() const noexcept { return _offset; }
    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] size_t byte_offset() const noexcept { return _offset * sizeof(T); }
    [[nodiscard]] size_t byte_size() const noexcept { return _size * sizeof(T); }
    
    [[nodiscard]] auto copy_from(const void *data) const { return [this, data](Dispatcher &d) { _copy_from(d, data); }; }
    [[nodiscard]] auto copy_to(void *data) const { return [this, data](Dispatcher &d) { _copy_to(d, data); }; }
    
    void clear_cache() const noexcept { _buffer->clear_cache(); }
    
    template<typename Modify, std::enable_if_t<std::is_invocable_v<Modify, T *>, int> = 0>
    [[nodiscard]] auto modify(Modify &&modify) {
        return [&modify, this](Dispatcher &dispatch) {
            _buffer->with_cache(dispatch, [&modify](void *raw_data) {
                modify(reinterpret_cast<T *>(raw_data));
            }, byte_offset(), byte_size());
        };
    }
    
    // For dsl
    template<typename Index>
    [[nodiscard]] auto operator[](Index &&index) const noexcept {
        using namespace luisa::compute::dsl;
        auto v = Variable::make_buffer_argument(type_desc<T>, _buffer);
        if (_offset == 0u) {
            auto i = Expr{std::forward<Index>(index)};
            return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, v, i.variable()))};
        } else {
            auto i = Expr{std::forward<Index>(index)} + static_cast<uint32_t>(_offset);
            return Expr<T>{Variable::make_temporary(type_desc<T>, std::make_unique<BinaryExpr>(BinaryOp::ACCESS, v, i.variable()))};
        }
    }
    
};

template<typename T>
inline auto Buffer::view(size_t offset, size_t size) noexcept {
    return BufferView<T>{shared_from_this(), offset, size};
}

}

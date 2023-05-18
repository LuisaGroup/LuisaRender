//
// Created by Mike on 2022/11/14.
//

#include <dsl/sugar.h>
#include <util/counter_buffer.h>

namespace luisa::render {

CounterBuffer::CounterBuffer(Device &device, uint size) noexcept
    : _buffer{device.create_buffer<uint2>(size)} {}

void CounterBuffer::record(Expr<uint> index, Expr<uint> count) noexcept {
    if (_buffer) {
        auto view = _buffer.view().as<uint>();
        auto old = view->atomic(index * 2u + 0u).fetch_add(count);
        $if(count != 0u & (old + count < old)) { view->atomic(index * 2u + 1u).fetch_add(1u); };
    }
}

void CounterBuffer::clear(Expr<uint> index) noexcept {
    if (_buffer) {
        auto view = _buffer.view().as<uint>();
        view->write(index * 2u + 0u, 0u);
        view->write(index * 2u + 1u, 0u);
    }
}

size_t CounterBuffer::size() const noexcept {
    return _buffer ? _buffer.size() / 2u : 0u;
}

luisa::unique_ptr<Command> CounterBuffer::copy_to(void *data) const noexcept {
    return _buffer ? _buffer.copy_to(data) : nullptr;
}

CounterBuffer::operator bool() const noexcept {
    return static_cast<bool>(_buffer);
}

}// namespace luisa::render

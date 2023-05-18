//
// Created by Mike on 2022/11/14.
//

#pragma once

#include <runtime/buffer.h>
#include <runtime/device.h>
#include <dsl/expr.h>

namespace luisa::render {

using compute::Buffer;
using compute::Command;
using compute::Device;
using compute::Expr;

class CounterBuffer {

private:
    Buffer<uint2> _buffer;

public:
    CounterBuffer() noexcept = default;
    CounterBuffer(Device &device, uint size) noexcept;
    void record(Expr<uint> index, Expr<uint> count = 1u) noexcept;
    void clear(Expr<uint> index) noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] luisa::unique_ptr<Command> copy_to(void *data) const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
};

}// namespace luisa::render

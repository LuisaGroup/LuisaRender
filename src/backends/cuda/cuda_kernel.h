//
// Created by Mike on 9/1/2020.
//

#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <type_traits>
#include <variant>

#include <cuda.h>

#include <compute/function.h>
#include <compute/kernel.h>
#include <core/logging.h>
#include <core/platform.h>

#include "cuda_check.h"

namespace luisa::cuda {

using luisa::compute::Kernel;
using luisa::compute::dsl::Variable;

struct ArgumentBufferView {

    void *buffer{nullptr};
    bool initialized{false};

    ArgumentBufferView(void *buffer, bool initialized) noexcept : buffer{buffer}, initialized{initialized} {}
};

class ArgumentBufferPool {

private:
    std::vector<void *> _allocated_buffers;
    std::vector<ArgumentBufferView> _available_buffers;
    size_t _unaligned_size{};
    size_t _aligned_size{};
    size_t _buffer_size{};
    std::mutex _mutex;

public:
    ~ArgumentBufferPool() noexcept;
    [[nodiscard]] ArgumentBufferView obtain() noexcept;
    void recycle(ArgumentBufferView buffer) noexcept;
    void create(size_t length, size_t alignment) noexcept;
};

class ArgumentEncoder {

public:
    struct Immutable {

        std::vector<std::byte> data;
        uint32_t argument_offset;

        Immutable(std::vector<std::byte> data, size_t offset) noexcept
            : data{std::move(data)}, argument_offset{static_cast<uint32_t>(offset)} {}
    };

    struct Uniform {

        const void *data;
        uint32_t data_size;
        uint32_t argument_offset;

        Uniform(const void *data, size_t data_size, size_t argument_offset) noexcept
            : data{data}, data_size{static_cast<uint32_t>(data_size)}, argument_offset{static_cast<uint32_t>(argument_offset)} {}
    };

private:
    std::vector<Immutable> _immutable_arguments;
    std::vector<Uniform> _uniform_bindings;
    size_t _encoded_length{};
    size_t _alignment{};

public:
    explicit ArgumentEncoder(const std::vector<Variable> &arguments);
    void encode(ArgumentBufferView &buffer) noexcept;
    void operator()(ArgumentBufferView &buffer) noexcept { encode(buffer); }
    [[nodiscard]] size_t encoded_length() const noexcept { return _encoded_length; }
    [[nodiscard]] size_t alignment() const noexcept { return _alignment; }
};

class CudaKernel : public Kernel {

private:
    CUfunction _handle;
    ArgumentEncoder _encode;
    ArgumentBufferPool _argument_buffer_pool;

protected:
    void _dispatch(compute::Dispatcher &dispatcher, uint2 blocks, uint2 block_size) override;

public:
    CudaKernel(CUfunction handle, ArgumentEncoder arg_enc) noexcept;
};

}// namespace luisa::cuda

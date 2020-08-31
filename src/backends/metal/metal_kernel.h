//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include "core/logging.h"
#include "core/platform.h"
#include <algorithm>
#import <condition_variable>
#import <memory>
#include <mutex>
#import <type_traits>
#import <variant>

#import <Metal/Metal.h>
#import <compute/kernel.h>

namespace luisa::metal {

using compute::Kernel;

struct ArgumentBufferView {

    id<MTLBuffer> handle{nullptr};
    uint32_t offset{0u};
    bool initialized{false};

    ArgumentBufferView(id<MTLBuffer> handle, size_t offset, bool initialized) noexcept
        : handle{handle}, offset{static_cast<uint32_t>(offset)}, initialized{initialized} {}
};

class ArgumentBufferPool {

private:
    id<MTLDevice> _device;
    std::vector<ArgumentBufferView> _buffers;
    size_t _size{};
    size_t _aligned_size{};
    size_t _buffer_size{};
    std::mutex _mutex;

public:
    ArgumentBufferPool(id<MTLDevice> device, size_t length, size_t alignment) noexcept
        : _device{device},
          _size{length},
          _aligned_size{(length + alignment - 1u) / alignment * alignment} {
        _buffer_size = std::max(_aligned_size, memory_page_size());
    }

    [[nodiscard]] ArgumentBufferView obtain() noexcept {
        std::lock_guard lock{_mutex};
        if (_buffers.empty()) {
            auto buffer = [_device newBufferWithLength:_buffer_size
                                               options:MTLResourceCPUCacheModeWriteCombined | MTLResourceHazardTrackingModeUntracked];
            for (auto offset = 0u; offset + _size <= _buffer_size; offset += _aligned_size) {
                _buffers.emplace_back(buffer, offset, false);
            }
        }
        auto arg_buffer = _buffers.back();
        _buffers.pop_back();
        return arg_buffer;
    }

    void recycle(ArgumentBufferView buffer) noexcept {
        std::lock_guard lock{_mutex};
        _buffers.emplace_back(std::move(buffer));
    }
};

class MetalKernel : public Kernel {

public:
    struct Uniform {
        uint32_t index{0u};
        uint32_t size{0u};
        const void *src{nullptr};
    };

    struct ImmutableArgument {
        std::vector<std::byte> data;
        uint32_t index{0u};
    };

    struct BufferArgument {
        id<MTLBuffer> handle{nullptr};
        MTLResourceUsage usage{};
        uint32_t offset{0u};
        uint32_t index{0u};
    };

    struct TextureArgument {
        id<MTLTexture> handle{nullptr};
        MTLResourceUsage usage{};
        uint32_t index{0u};
    };

    using Argument = std::variant<ImmutableArgument, BufferArgument, TextureArgument>;

private:
    id<MTLComputePipelineState> _handle;
    std::vector<Uniform> _argument_bindings;
    std::vector<Argument> _arguments;
    id<MTLArgumentEncoder> _argument_encoder;
    ArgumentBufferPool _argument_buffer_pool;
    id<MTLBuffer> _constant_argument_buffer{nullptr};

    [[nodiscard]] ArgumentBufferView _get_argument_buffer();
    void _initialize_argument_buffer(id<MTLBuffer> buffer, size_t offset);

protected:
    void _dispatch(compute::Dispatcher &dispatcher, uint2 threadgroups, uint2 threadgroup_size) override;

public:
    MetalKernel(
        id<MTLComputePipelineState> handle,
        std::vector<Uniform> uniforms,
        std::vector<Argument> args,
        id<MTLArgumentEncoder> arg_enc) noexcept;
    [[nodiscard]] id<MTLComputePipelineState> handle() const noexcept { return _handle; }
};

}

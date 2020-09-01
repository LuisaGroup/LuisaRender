//
// Created by Mike on 9/1/2020.
//

#include "cuda_kernel.h"
#include "cuda_buffer.h"
#include "cuda_texture.h"
#include "cuda_dispatcher.h"

namespace luisa::cuda {

ArgumentBufferView ArgumentBufferPool::obtain() noexcept {
    std::lock_guard lock{_mutex};
    if (_allocated_buffers.empty()) {
        void *buffer = nullptr;
        CUDA_CHECK(cuMemHostAlloc(&buffer, _buffer_size, 0));
        _allocated_buffers.emplace_back(buffer);
        for (auto offset = 0u; offset + _unaligned_size <= _buffer_size; offset += _aligned_size) {
            _available_buffers.emplace_back(reinterpret_cast<std::byte *>(buffer) + offset, false);
        }
    }
    auto arg_buffer = _available_buffers.back();
    _available_buffers.pop_back();
    return arg_buffer;
}

void ArgumentBufferPool::recycle(ArgumentBufferView buffer) noexcept {
    std::lock_guard lock{_mutex};
    _available_buffers.emplace_back(buffer);
}

ArgumentBufferPool::~ArgumentBufferPool() noexcept {
    for (auto p : _allocated_buffers) {
        CUDA_CHECK(cuMemFreeHost(p));
    }
}

void ArgumentBufferPool::create(size_t length, size_t alignment) noexcept {
    _unaligned_size = length;
    _aligned_size = (length + alignment - 1u) / alignment * alignment;
    _buffer_size = std::max(_aligned_size, memory_page_size());
}

ArgumentEncoder::ArgumentEncoder(const std::vector<Variable> &arguments) : _alignment{16u} {

    auto offset = 0u;
    auto align_offset = [&offset](size_t alignment) noexcept {
        return offset = (offset + alignment - 1u) / alignment * alignment;
    };

    for (auto &&arg : arguments) {
        if (arg.is_buffer_argument()) {
            auto buffer_view = arg.buffer();
            auto device_ptr = dynamic_cast<CudaBuffer *>(buffer_view.buffer())->handle() + buffer_view.byte_offset();
            auto size = 8u;
            auto alignment = 8u;
            std::vector<std::byte> bytes(size);
            reinterpret_cast<CUdeviceptr *>(bytes.data())[0] = device_ptr;
            _immutable_arguments.emplace_back(std::move(bytes), align_offset(alignment));
            LUISA_INFO("Buffer offset: ", offset);
            offset += size;
        } else if (arg.is_texture_argument()) {
            auto alignment = 8u;
            auto size = 16u;
            auto texture = dynamic_cast<CudaTexture *>(arg.texture())->texture_handle();
            auto surface = dynamic_cast<CudaTexture *>(arg.texture())->surface_handle();
            std::vector<std::byte> bytes(size);
            reinterpret_cast<CUtexObject *>(bytes.data())[0] = texture;
            reinterpret_cast<CUsurfObject *>(bytes.data())[1] = surface;
            _immutable_arguments.emplace_back(std::move(bytes), align_offset(alignment));
            LUISA_INFO("Texture offset: ", offset);
            offset += size;
        } else if (arg.is_immutable_argument()) {
            auto alignment = arg.type()->alignment;
            _immutable_arguments.emplace_back(arg.immutable_data(), align_offset(alignment));
            LUISA_INFO("Immutable data alignment: ", alignment, ", offset: ", offset);
            offset += arg.immutable_data().size();
        } else if (arg.is_uniform_argument()) {
            auto alignment = arg.type()->alignment;
            _uniform_bindings.emplace_back(arg.uniform_data(), arg.type()->size, align_offset(alignment));
            LUISA_INFO("Uniform data alignment: ", alignment, ", offset: ", offset);
            offset += arg.immutable_data().size();
        } else {
            LUISA_EXCEPTION("Unsupported argument type.");
        }
    }
    _encoded_length = align_offset(_alignment);
}

void ArgumentEncoder::encode(ArgumentBufferView &buffer) noexcept {
    auto buffer_base = reinterpret_cast<std::byte *>(buffer.buffer);
    if (!buffer.initialized) {
        for (auto &&immutable : _immutable_arguments) {
            std::memmove(buffer_base + immutable.argument_offset, immutable.data.data(), immutable.data.size());
        }
        buffer.initialized = true;
    }
    for (auto &&uniform : _uniform_bindings) {
        std::memmove(buffer_base + uniform.argument_offset, uniform.data, uniform.data_size);
    }
}

CudaKernel::CudaKernel(CUfunction handle, ArgumentEncoder arg_enc) noexcept
    : _handle{handle}, _encode{std::move(arg_enc)} {
    _argument_buffer_pool.create(_encode.encoded_length(), _encode.alignment());
}

void CudaKernel::_dispatch(compute::Dispatcher &dispatcher, uint2 blocks, uint2 block_size) {
    auto argument_buffer = _argument_buffer_pool.obtain();
    _encode(argument_buffer);
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    CUDA_CHECK(cuLaunchKernel(_handle, blocks.x, blocks.y, 1u, block_size.x, block_size.y, 1u, 0u, stream, &argument_buffer.buffer, nullptr));
}

}// namespace luisa::cuda

//
// Created by Mike on 9/1/2020.
//

#include "cuda_kernel.h"
#include "cuda_buffer.h"
#include "cuda_texture.h"
#include "cuda_dispatcher.h"

namespace luisa::cuda {

void CudaKernel::_dispatch(compute::Dispatcher &dispatcher, uint2 blocks, uint2 block_size) {
    
    if (!_uniforms.empty()) {
        for (auto &&u : _uniforms) {
            if (u.binding != nullptr) {
                std::memmove(_arguments.data() + u.offset, u.binding, u.binding_size);
            } else {
                std::memmove(_arguments.data() + u.offset, u.immutable.data(), u.immutable.size());
            }
        }
    }
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    void *args = _arguments.data();
    CUDA_CHECK(cuLaunchKernel(_handle, blocks.x, blocks.y, 1u, block_size.x, block_size.y, 1u, 0u, stream, &args, nullptr));
}

CudaKernel::CudaKernel(CUfunction handle, std::vector<Kernel::Resource> resources, std::vector<Kernel::Uniform> uniforms) noexcept
    : Kernel{std::move(resources), std::move(uniforms)}, _handle{handle} {
    
    size_t res_offset = 0u;
    if (!_uniforms.empty()) {
        res_offset = _uniforms.back().offset + std::max(_uniforms.back().binding_size, _uniforms.back().immutable.size());
    }
    res_offset = (res_offset + 7u) / 8u * 8u;
    auto size = res_offset;
    for (auto &&res : _resources) { size += res.buffer != nullptr ? 8u : 16u; }
    _arguments.resize(size);
    for (auto &&res : _resources) {
        if (res.buffer != nullptr) {
            auto cuda_buffer = dynamic_cast<CudaBuffer *>(res.buffer.get())->handle();
            std::memmove(_arguments.data() + res_offset, &cuda_buffer, 8u);
            res_offset += 8u;
        } else {
            auto cuda_texture = dynamic_cast<CudaTexture *>(res.texture.get())->texture_handle();
            auto cuda_surface = dynamic_cast<CudaTexture *>(res.texture.get())->surface_handle();
            std::memmove(_arguments.data() + res_offset, &cuda_texture, 8u);
            std::memmove(_arguments.data() + res_offset + 8u, &cuda_surface, 8u);
            res_offset += 16u;
        }
    }
}
    
}// namespace luisa::cuda

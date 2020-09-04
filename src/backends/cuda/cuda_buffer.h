//
// Created by Mike on 8/30/2020.
//

#pragma once

#include <cuda.h>
#include <compute/buffer.h>

#include "cuda_host_cache.h"

namespace luisa::cuda {

using luisa::compute::Buffer;

class CudaBuffer : public Buffer {

private:
    CUdeviceptr _handle;
    CudaHostCache _host_cache;

public:
    CudaBuffer(CUdeviceptr handle, size_t size) noexcept : Buffer{size}, _handle{handle}, _host_cache{size} {}
    ~CudaBuffer() noexcept override;
    void upload(compute::Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) override;
    void download(compute::Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) override;
    void clear_cache() noexcept override { _host_cache.clear(); }
    [[nodiscard]] CUdeviceptr handle() const noexcept { return _handle; }
    void with_cache(compute::Dispatcher &dispatch, const std::function<void(void *)> &modify, size_t offset, size_t length) override;
};

}

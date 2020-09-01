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
    CudaBuffer(CUdeviceptr handle, size_t size, size_t host_caches) noexcept : Buffer{size, host_caches}, _handle{handle}, _host_cache{size, host_caches} {}
    ~CudaBuffer() noexcept override;
    void upload(compute::Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) override;
    void download(compute::Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) override;
    void clear_cache() noexcept override { _host_cache.clear(); }
    [[nodiscard]] CUdeviceptr handle() const noexcept { return _handle; }
};



}

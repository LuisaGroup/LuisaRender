//
// Created by Mike on 8/30/2020.
//

#include "cuda_buffer.h"
#include "cuda_check.h"
#include "cuda_dispatcher.h"

namespace luisa::cuda {

void CudaBuffer::upload(compute::Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) {
    auto cache = _host_cache.obtain();
    std::memmove(cache, host_data, size);
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    CUDA_CHECK(cuMemcpyAsync(_handle + offset, reinterpret_cast<CUdeviceptr>(cache), size, stream));
    dispatcher.when_completed([cache, this] { _host_cache.recycle(cache); });
}

void CudaBuffer::download(compute::Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) {
    auto cache = _host_cache.obtain();
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    CUDA_CHECK(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(cache), _handle + offset, size, stream));
    dispatcher.when_completed([cache, this, dst = host_buffer, size] {
        std::memmove(dst, cache, size);
        _host_cache.recycle(cache);
    });
}

CudaBuffer::~CudaBuffer() noexcept {
    CUDA_CHECK(cuMemFree(_handle));
    _host_cache.clear();
}

}

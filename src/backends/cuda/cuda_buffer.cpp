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
    CUDA_CHECK(cuMemcpyHtoDAsync(_handle + offset, cache, size, stream));
    dispatcher.when_completed([cache, this] { _host_cache.recycle(cache); });
}

void CudaBuffer::download(compute::Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) {
    auto stream = dynamic_cast<CudaDispatcher &>(dispatcher).handle();
    CUDA_CHECK(cuMemcpyDtoHAsync(host_buffer, _handle + offset, size, stream));
}

CudaBuffer::~CudaBuffer() noexcept {
    CUDA_CHECK(cuMemFree(_handle));
    _host_cache.clear();
}

void CudaBuffer::with_cache(Dispatcher &dispatch, const std::function<void(void *)> &modify, size_t offset, size_t length) {
    auto cache = _host_cache.obtain();
    modify(cache);
    auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
    CUDA_CHECK(cuMemcpyAsync(_handle + offset, reinterpret_cast<CUdeviceptr>(cache), length, stream));
    dispatch.when_completed([cache, this] { _host_cache.recycle(cache); });
}

}

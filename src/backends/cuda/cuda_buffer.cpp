//
// Created by Mike on 8/30/2020.
//

#include "cuda_buffer.h"

namespace luisa::cuda {

void CudaBuffer::upload(compute::Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) {
}

void CudaBuffer::download(compute::Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) {
}

void CudaBuffer::clear_cache() noexcept {
}

}

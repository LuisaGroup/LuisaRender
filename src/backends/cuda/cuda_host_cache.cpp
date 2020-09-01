//
// Created by Mike on 9/1/2020.
//

#include <core/platform.h>

#include "cuda_check.h"
#include "cuda_host_cache.h"

namespace luisa::cuda {

void *CudaHostCache::obtain() noexcept {
    std::lock_guard lock{_mutex};
    void *cache = nullptr;
    if (_available_caches.empty()) {
        CUDA_CHECK(cuMemHostAlloc(&cache, _size, 0));
        LUISA_INFO("Created host cache buffer #", _allocated_caches.size(), " with length ", _size, " for device content synchronization.");
        _allocated_caches.emplace(cache);
    } else {
        cache = _available_caches.back();
        _available_caches.pop_back();
    }
    return cache;
}

CudaHostCache::CudaHostCache(size_t size) noexcept
    : _size{size} {}

void CudaHostCache::recycle(void *cache) noexcept {
    std::lock_guard lock{_mutex};
    LUISA_EXCEPTION_IF(_allocated_caches.find(cache) == _allocated_caches.end(), "Recycled cache is not allocated by CudaHostCache.");
    _available_caches.emplace_back(cache);
}

void CudaHostCache::clear() noexcept {
    std::lock_guard lock{_mutex};
    for (auto p : _allocated_caches) { CUDA_CHECK(cuMemFreeHost(p)); }
    _allocated_caches.clear();
    _available_caches.clear();
}

}// namespace luisa::cuda

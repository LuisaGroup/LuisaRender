//
// Created by Mike on 9/1/2020.
//

#include <core/platform.h>

#include "check.h"
#include "cuda_host_cache.h"

namespace luisa::cuda {

void *CudaHostCache::obtain() noexcept {
    std::unique_lock lock{_mutex};
    _cv.wait(lock, [this] { return !_available_caches.empty(); });
    auto cache = _available_caches.back();
    lock.unlock();
    if (cache == nullptr) {
        CUDA_CHECK(cuMemAllocHost(&cache, _size));
        LUISA_INFO("Created host cache buffer #", _count++, " with length ", _size, " for device content synchronization.");
    }
    return cache;
}

CudaHostCache::CudaHostCache(size_t size, size_t max_count) noexcept
    : _size{size}, _max_count{max_count} {
    _available_caches.resize(max_count, nullptr);
}

void CudaHostCache::recycle(void *cache) noexcept {
    {
        std::lock_guard lock{_mutex};
        _available_caches.emplace_back(cache);
    }
    _cv.notify_one();
}

void CudaHostCache::clear() noexcept {
    std::unique_lock lock{_mutex};
    _cv.wait(lock, [this] { return _available_caches.size() == _max_count; });
    for (auto p : _available_caches) {
        if (p != nullptr) { CUDA_CHECK(cuMemFreeHost(p)); }
    }
    _available_caches.clear();
    _available_caches.resize(_max_count, nullptr);
}

}// namespace luisa::cuda

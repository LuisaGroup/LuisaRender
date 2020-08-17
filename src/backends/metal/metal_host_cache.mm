//
// Created by Mike Smith on 2020/8/17.
//

#import <core/logging.h>
#import "metal_host_cache.h"

namespace luisa::metal {

MetalHostCache::MetalHostCache(id<MTLDevice> device, size_t size, size_t count) noexcept
    : _device{device}, _cache_size{size}, _max_count{count} {
    _available_caches.resize(count, nullptr);
}

id<MTLBuffer> MetalHostCache::get() noexcept {
    std::unique_lock lock{_cache_mutex};
    _cache_cv.wait(lock, [this] { return !_available_caches.empty(); });
    id<MTLBuffer> cache = _available_caches.back();
    _available_caches.pop_back();
    lock.unlock();
    if (cache == nullptr) {
        cache = [_device newBufferWithLength:_cache_size options:MTLResourceStorageModeShared];
        LUISA_INFO("Created host cache buffer #", _cache_count++, " with length ", _cache_size, " for device content synchronization.");
    }
    return cache;
}

void MetalHostCache::recycle(id<MTLBuffer> cache) noexcept {
    {
        std::lock_guard lock{_cache_mutex};
        _available_caches.emplace_back(cache);
    }
    _cache_cv.notify_one();
}

void MetalHostCache::clear() noexcept {
    std::unique_lock lock{_cache_mutex};
    _cache_cv.wait(lock, [this] { return _available_caches.size() == _max_count; });
    _available_caches.clear();
    _available_caches.resize(_max_count, nullptr);
}

}

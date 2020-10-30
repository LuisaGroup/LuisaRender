//
// Created by Mike Smith on 2020/8/17.
//

#import "metal_host_cache.h"
#import <core/logging.h>

namespace luisa::metal {

MetalHostCache::MetalHostCache(id<MTLDevice> device, size_t size) noexcept
    : _device{device}, _cache_size{size} {}

id<MTLBuffer> MetalHostCache::obtain() noexcept {
    std::scoped_lock lock{_cache_mutex};
    id<MTLBuffer> cache = nullptr;
    if (_available_caches.empty()) {
        cache = [_device newBufferWithLength:_cache_size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        _allocated_caches.emplace(cache);
    } else {
        cache = _available_caches.back();
        _available_caches.pop_back();
    }
    return cache;
}

void MetalHostCache::recycle(id<MTLBuffer> cache) noexcept {
    std::scoped_lock lock{_cache_mutex};
    LUISA_EXCEPTION_IF(_allocated_caches.find(cache) == _allocated_caches.cend(), "Recycled cache is not allocated by MetalHostCache.");
    _available_caches.emplace_back(cache);
}

void MetalHostCache::clear() noexcept {
    std::scoped_lock lock{_cache_mutex};
    _available_caches.clear();
    _allocated_caches.clear();
}

}

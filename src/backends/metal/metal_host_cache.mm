//
// Created by Mike Smith on 2020/8/17.
//

#import "metal_host_cache.h"
#import <core/logging.h>

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
        using namespace std::chrono_literals;
        auto tt0 = std::chrono::high_resolution_clock::now();
        _allocated_memory.emplace_back(_cache_size);
        auto &&memory = _allocated_memory.back();
        auto t0 = std::chrono::high_resolution_clock::now();
        LUISA_INFO("Time spent on PageAlignedBuffer::ctor = ", (t0 - tt0) / 1ns, "ns");
        cache = [_device newBufferWithBytesNoCopy:memory.data()
                                           length:memory.aligned_byte_size()
                                          options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                      deallocator:nullptr];
        auto t1 = std::chrono::high_resolution_clock::now();
        LUISA_INFO("Time spent on MTLDevice::newBufferWithBytesNoCopy = ", (t1 - t0) / 1ns, "ns");
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

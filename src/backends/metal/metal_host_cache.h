//
// Created by Mike Smith on 2020/8/17.
//

#pragma once

#import <set>
#import <vector>
#import <mutex>
#import <condition_variable>

#import <Metal/Metal.h>
#import <core/concepts.h>

namespace luisa::metal {

class MetalHostCache : Noncopyable {

private:
    id<MTLDevice> _device;
    std::set<id<MTLBuffer>> _allocated_caches;
    std::vector<id<MTLBuffer>> _available_caches;
    std::mutex _cache_mutex;
    size_t _cache_size{0u};

public:
    MetalHostCache(id<MTLDevice> device, size_t size) noexcept;
    ~MetalHostCache() noexcept { clear(); }
    [[nodiscard]] id<MTLBuffer> obtain() noexcept;
    void recycle(id<MTLBuffer> cache) noexcept;
    void clear() noexcept;
};

}

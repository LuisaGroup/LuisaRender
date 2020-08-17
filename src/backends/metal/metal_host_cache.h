//
// Created by Mike Smith on 2020/8/17.
//

#pragma once

#import <vector>
#import <mutex>
#import <condition_variable>

#import <Metal/Metal.h>
#import <core/concepts.h>

namespace luisa::metal {

class MetalHostCache : Noncopyable {

private:
    id<MTLDevice> _device;
    std::vector<id<MTLBuffer>> _available_caches;
    std::mutex _cache_mutex;
    std::condition_variable _cache_cv;
    size_t _cache_count{0u};
    size_t _cache_size{0u};
    size_t _max_count{0u};

public:
    MetalHostCache(id<MTLDevice> device, size_t size, size_t count) noexcept;
    ~MetalHostCache() noexcept { clear(); }
    [[nodiscard]] id<MTLBuffer> get() noexcept;
    void recycle(id<MTLBuffer> cache) noexcept;
    void clear() noexcept;
};

}

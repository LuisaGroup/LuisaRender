//
// Created by Mike on 9/1/2020.
//

#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>

#include <cuda.h>

namespace luisa::cuda {

class CudaHostCache {

private:
    std::vector<void *> _available_caches;
    std::mutex _mutex;
    std::condition_variable _cv;
    size_t _count{0u};
    size_t _size;
    size_t _max_count;

public:
    CudaHostCache(size_t size, size_t max_count) noexcept;
    ~CudaHostCache() noexcept {
        LUISA_INFO("CudaHostCache: ", 0);
        clear();
        LUISA_INFO("CudaHostCache: ", 1);
    }
    
    [[nodiscard]] void *obtain() noexcept;
    void recycle(void *cache) noexcept;
    void clear() noexcept;
};

}

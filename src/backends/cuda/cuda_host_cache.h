//
// Created by Mike on 9/1/2020.
//

#pragma once

#include <set>
#include <vector>
#include <mutex>
#include <condition_variable>

#include <cuda.h>

namespace luisa::cuda {

class CudaHostCache {

private:
    std::set<void *> _allocated_caches;
    std::vector<void *> _available_caches;
    std::mutex _mutex;
    size_t _size;

public:
    explicit CudaHostCache(size_t size) noexcept;
    ~CudaHostCache() noexcept { clear(); }
    
    [[nodiscard]] void *obtain() noexcept;
    void recycle(void *cache) noexcept;
    void clear() noexcept;
};

}

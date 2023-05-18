//
// Created by Mike Smith on 2023/5/18.
//

#include <util/thread_pool.h>

namespace luisa::render {

ThreadPool &global_thread_pool() noexcept {
    static ThreadPool pool{std::thread::hardware_concurrency()};
    return pool;
}

}// namespace luisa::render

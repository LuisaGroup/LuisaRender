//
// Created by Mike Smith on 2023/5/18.
//

#include <core/thread_pool.h>

namespace luisa::render {

[[nodiscard]] ThreadPool &global_thread_pool() noexcept;

}// namespace luisa::render
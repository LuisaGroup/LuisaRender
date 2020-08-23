//
// Created by Mike Smith on 2020/8/14.
//

#include "kernel.h"
#include "dispatcher.h"

namespace luisa::compute {

void Kernel::dispatch(Dispatcher &dispatcher, uint threads, uint threadgroup_size) {
    _dispatch(dispatcher, make_uint2((threads + threadgroup_size - 1u) / threadgroup_size, 1u), make_uint2(threadgroup_size, 1u));
}

void Kernel::dispatch(Dispatcher &dispatcher, uint2 threads, uint2 threadgroup_size) {
    _dispatch(dispatcher, (threads + threadgroup_size - 1u) / threadgroup_size, threadgroup_size);
}

}

//
// Created by Mike Smith on 2020/8/14.
//

#include "kernel.h"

namespace luisa::compute {

void Kernel::dispatch(Dispatcher &dispatcher, uint threadgroups, uint threadgroup_size) {
    dispatch(dispatcher, make_uint3(threadgroups, 1u, 1u), make_uint3(threadgroup_size, 1u, 1u));
}

void Kernel::dispatch(Dispatcher &dispatcher, uint2 threadgroups, uint2 threadgroup_size) {
    dispatch(dispatcher, make_uint3(threadgroups, 1u), make_uint3(threadgroup_size, 1u));
}

}

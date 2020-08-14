//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <vector>
#include <tuple>

#include <core/data_types.h>
#include <compute/buffer.h>
#include <compute/dispatcher.h>

namespace luisa::compute {

struct Kernel : Noncopyable {
    virtual ~Kernel() noexcept = default;
    virtual void dispatch(Dispatcher &dispatcher, uint threadgroups, uint threadgroup_size);
    virtual void dispatch(Dispatcher &dispatcher, uint2 threadgroups, uint2 threadgroup_size);
    virtual void dispatch(Dispatcher &dispatcher, uint3 threadgroups, uint3 threadgroup_size) = 0;
};

}

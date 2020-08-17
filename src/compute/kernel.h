//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <vector>
#include <tuple>

#include <core/data_types.h>
#include <compute/buffer.h>

namespace luisa::compute {

class Dispatcher;

class Kernel : Noncopyable {

protected:
    virtual void _dispatch(Dispatcher &dispatcher, uint3 threadgroups, uint3 threadgroup_size) = 0;
    
public:
    virtual ~Kernel() noexcept = default;
    virtual void dispatch(Dispatcher &dispatcher, uint threads, uint threadgroup_size);
    virtual void dispatch(Dispatcher &dispatcher, uint2 threads, uint2 threadgroup_size);
    virtual void dispatch(Dispatcher &dispatcher, uint3 threads, uint3 threadgroup_size);
};

}

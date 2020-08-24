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
    virtual void _dispatch(Dispatcher &dispatcher, uint2 threadgroups, uint2 threadgroup_size) = 0;
    
public:
    virtual ~Kernel() noexcept = default;
    
    [[nodiscard]] auto parallelize(uint threads, uint threadgroup_size = 128u) {
        return [this, threads, tg_size = threadgroup_size](Dispatcher &dispatch) {
            _dispatch(dispatch, make_uint2((threads + tg_size - 1u) / tg_size, 1u), make_uint2(tg_size, 1u));
        };
    }
    
    [[nodiscard]] auto parallelize(uint2 threads, uint2 threadgroup_size = make_uint2(8u, 8u)) {
        return [this, threads, tg_size = threadgroup_size](Dispatcher &dispatch) {
            _dispatch(dispatch, (threads + tg_size - 1u) / tg_size, tg_size);
        };
    }
};

}

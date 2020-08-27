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

class LUISA_EXPORT Kernel : Noncopyable {

protected:
    virtual void _dispatch(Dispatcher &dispatcher, uint2 blocks, uint2 block_size) = 0;
    
public:
    virtual ~Kernel() noexcept = default;
    
    [[nodiscard]] auto parallelize(uint threads, uint block_size = 128u) {
        return [this, threads, block_size](Dispatcher &dispatch) {
            _dispatch(dispatch, make_uint2((threads + block_size - 1u) / block_size, 1u), make_uint2(block_size, 1u));
        };
    }
    
    [[nodiscard]] auto parallelize(uint2 threads, uint2 block_size = make_uint2(8u, 8u)) {
        return [this, threads, block_size](Dispatcher &dispatch) {
            _dispatch(dispatch, (threads + block_size - 1u) / block_size, block_size);
        };
    }
};

}

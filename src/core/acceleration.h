//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#include <compute/kernel.h>
#include "ray.h"
#include "hit.h"

namespace luisa {

struct Acceleration : Noncopyable {
    virtual ~Acceleration() = default;
    virtual void refit(KernelDispatcher &dispatch) = 0;
    virtual void trace_any(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<AnyHit> its_buffer, BufferView<uint> ray_count_buffer) = 0;
    virtual void trace_closest(KernelDispatcher &dispatch, BufferView<Ray> ray_buffer, BufferView<ClosestHit> its_buffer, BufferView<uint> ray_count_buffer) = 0;
};

}

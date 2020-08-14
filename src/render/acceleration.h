//
// Created by Mike Smith on 2020/8/14.
//

#pragma once

#include <compute/dispatcher.h>
#include <render/ray.h>
#include <render/hit.h>

namespace luisa::render {

using compute::Dispatcher;
using compute::Buffer;
using compute::BufferView;

struct Acceleration {
    virtual ~Acceleration() noexcept = default;
    virtual void refit(Dispatcher &dispatcher) = 0;
    virtual void intersect_any(Dispatcher &dispatcher, BufferView<Ray> ray_buffer, BufferView<AnyHit> its_buffer, BufferView<uint> ray_count_buffer) = 0;
    virtual void intersect_closest(Dispatcher &dispatcher, BufferView<Ray> ray_buffer, BufferView<ClosestHit> its_buffer, BufferView<uint> ray_count_buffer) = 0;
};

}

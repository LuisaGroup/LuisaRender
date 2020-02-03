//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "data_types.h"
#include "node.h"
#include "kernel.h"
#include "ray.h"

namespace luisa {

class Film;

class Filter : public Node {

protected:
    float _radius;

public:
    Filter(Device *device, float radius) noexcept : Node{device}, _radius{radius} {}
    virtual void add_samples(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue, Film &film) = 0;
    [[nodiscard]] float radius() const noexcept { return _radius; }
};

}

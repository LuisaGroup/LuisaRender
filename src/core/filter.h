//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include "data_types.h"
#include "node.h"
#include "kernel.h"
#include "ray.h"
#include "parser.h"

namespace luisa {

class Film;

class Filter : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Filter);

protected:
    float _radius;

public:
    Filter(Device *device, const ParameterSet &parameters)
        : Node{device}, _radius{parameters["radius"].parse_float_or_default(1.0f)} {}
    virtual void add_samples(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue, Film &film) = 0;
    [[nodiscard]] float radius() const noexcept { return _radius; }
};

}

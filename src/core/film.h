//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "data_types.h"
#include "device.h"
#include "ray.h"
#include "node.h"

namespace luisa {

class Film : public Node {

protected:
    uint2 _size;

public:
    virtual void gather_rays(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue) = 0;
};

}

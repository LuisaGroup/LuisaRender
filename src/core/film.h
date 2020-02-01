//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "data_types.h"
#include "device.h"
#include "ray.h"

namespace luisa {

class Film {

private:
    uint2 _size;
    std::unique_ptr<Buffer> _pixel_;

public:
    virtual void generate_pixel_samples(KernelDispatcher &dispatch, RayPool &ray_pool) = 0;
    virtual void gather_rays(KernelDispatcher &dispatch, RayPool &ray_pool) = 0;
};

}

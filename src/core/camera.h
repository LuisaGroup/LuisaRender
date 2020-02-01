//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "ray.h"
#include "device.h"

namespace luisa {

class Camera {

private:
    Device *_device;

public:
    Camera(Device *device, size_t ray_queue_capacity) : _device{device} {}
    virtual void update(float time) = 0;
    virtual void generate_rays(KernelDispatcher &dispatch, RayPool &ray_pool) = 0;
};

}

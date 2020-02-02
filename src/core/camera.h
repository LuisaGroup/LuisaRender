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
    explicit Camera(Device *device) : _device{device} {}
    virtual void update(float time) = 0;
    virtual void generate_rays(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue) = 0;
};

}

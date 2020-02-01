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
    std::unique_ptr<Buffer> _ray_queue_buffer;
    std::unique_ptr<Buffer> _ray_queue_size_buffer;

public:
    Camera(Device *device, size_t ray_queue_capacity) : _device{device} {
        _ray_queue_buffer = _device->create_buffer(ray_queue_capacity * sizeof(uint), BufferStorage::DEVICE_PRIVATE);
        _ray_queue_size_buffer = _device->create_buffer(ray_queue_capacity * sizeof(uint), BufferStorage::DEVICE_PRIVATE);
    }
    virtual void update(float time) = 0;
    virtual void generate_rays(KernelDispatcher &dispatch, RayPool &ray_pool) = 0;
    Buffer &ray_queue_buffer() noexcept { return *_ray_queue_buffer; }
    Buffer &ray_queue_size_buffer() noexcept { return *_ray_queue_size_buffer; }
};

}

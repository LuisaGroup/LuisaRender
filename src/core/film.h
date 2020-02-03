//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include "data_types.h"
#include "device.h"
#include "ray.h"
#include "node.h"
#include "filter.h"

namespace luisa {

class Film : public Node {

protected:
    uint2 _resolution;
    std::unique_ptr<Buffer> _accumulation_buffer;
    std::shared_ptr<Filter> _filter;
    std::unique_ptr<Kernel> _clear_accumulation_buffer_kernel;

public:
    Film(Device *device, uint2 resolution, std::shared_ptr<Filter> filter)
        : Node{device}, _resolution{resolution}, _filter{std::move(filter)} {
        _accumulation_buffer = device->create_buffer<uint4>(resolution.x * resolution.y, BufferStorage::DEVICE_PRIVATE);
        _clear_accumulation_buffer_kernel = device->create_kernel("film_clear_accumulation_kernel");
    }
    
    void clear_accumulation_buffer(KernelDispatcher &dispatch) {
        dispatch(*_clear_accumulation_buffer_kernel, _resolution, [&](KernelArgumentEncoder &encode) {
            encode("accumulation_buffer", *_accumulation_buffer);
            encode("resolution", _resolution);
        });
    }
    
    virtual void gather_rays(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue) {
        _filter->add_samples(dispatch, ray_pool, ray_queue, *this);
    }
    virtual void postprocess(KernelDispatcher &dispatch) = 0;
    virtual void save(const std::filesystem::path &filename) = 0;
    [[nodiscard]] Filter &filter() noexcept { return *_filter; }
    [[nodiscard]] BufferView<uint4> accumulation_buffer() noexcept { return _accumulation_buffer->view<uint4>(); }
    [[nodiscard]] uint2 resolution() noexcept { return _resolution; }
};

}

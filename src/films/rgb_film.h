//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/film.h>

namespace luisa {

class RGBFilm : public Film {

private:
    std::unique_ptr<Buffer> _framebuffer;
    std::unique_ptr<Kernel> _postprocess_kernel;

public:
    RGBFilm(Device *device, uint2 resolution, std::shared_ptr<Filter> filter) : Film{device, resolution, std::move(filter)} {
        _framebuffer = device->create_buffer<float3>(resolution.x * resolution.y, BufferStorage::DEVICE_PRIVATE);
        _postprocess_kernel = device->create_kernel("rgb_film_postprocess_kernel");
    }
    void postprocess(KernelDispatcher &dispatch) override;
    void save(const std::filesystem::path &filename) override;
    
};

}

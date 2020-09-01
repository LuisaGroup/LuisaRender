//
// Created by Mike on 9/1/2020.
//

#pragma once

#include <cuda.h>

#include <compute/texture.h>
#include "cuda_check.h"
#include "cuda_host_cache.h"

namespace luisa::cuda {

using compute::Texture;
using compute::Buffer;
using compute::Dispatcher;
using compute::PixelFormat;

class CudaTexture : public Texture {

private:
    CUarray _array_handle;
    CUtexObject _texture_handle;
    CUsurfObject _surface_handle;
    CudaHostCache _cache;

protected:
    void _copy_from(Dispatcher &dispatcher, Buffer *buffer, size_t offset) override;
    void _copy_to(Dispatcher &dispatcher, Buffer *buffer, size_t offset) override;
    void _copy_to(Dispatcher &dispatcher, Texture *texture) override;
    void _copy_from(Dispatcher &dispatcher, const void *data) override;
    void _copy_to(Dispatcher &dispatcher, void *data) override;

public:
    CudaTexture(CUarray array_handle, CUtexObject tex_handle, CUsurfObject surf_handle, uint32_t width, uint32_t height, PixelFormat format) noexcept
        : Texture{width, height, format},
          _array_handle{array_handle},
          _texture_handle{tex_handle},
          _surface_handle{surf_handle},
          _cache{byte_size()} {}
    
    ~CudaTexture() noexcept override {
        CUDA_CHECK(cuArrayDestroy(_array_handle));
        CUDA_CHECK(cuTexObjectDestroy(_texture_handle));
        CUDA_CHECK(cuSurfObjectDestroy(_surface_handle));
    }
    
    [[nodiscard]] CUarray array_handle() const noexcept { return _array_handle; }
    [[nodiscard]] CUtexObject texture_handle() const noexcept { return _texture_handle; }
    [[nodiscard]] CUsurfObject surface_handle() const noexcept { return _surface_handle; }
    void clear_cache() override { _cache.clear(); }
};

}

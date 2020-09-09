//
// Created by Mike Smith on 2020/8/16.
//

#pragma once

#import <compute/texture.h>
#import "metal_host_cache.h"

namespace luisa::metal {

using compute::Texture;
using compute::Buffer;
using compute::Dispatcher;

class MetalTexture : public Texture {

private:
    id<MTLTexture> _handle;
    MetalHostCache _cache;

public:
    MetalTexture(id<MTLTexture> handle, uint32_t width, uint32_t height, compute::PixelFormat format) noexcept
        : Texture{width, height, format}, _handle{handle}, _cache{[handle device], byte_size()} {}
    
    [[nodiscard]] id<MTLTexture> handle() const noexcept { return _handle; }
    void clear_cache() override;

    void copy_from(Dispatcher &dispatcher, Buffer *buffer, size_t offset) override;
    void copy_to(Dispatcher &dispatcher, Buffer *buffer, size_t offset) override;
    void copy_to(Dispatcher &dispatcher, Texture *texture) override;
    void copy_from(Dispatcher &dispatcher, const void *data) override;
    void copy_to(Dispatcher &dispatcher, void *data) override;
};

}

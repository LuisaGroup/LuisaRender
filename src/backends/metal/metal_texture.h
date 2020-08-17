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

protected:
    void _copy_from(Dispatcher &dispatcher, Buffer *buffer, size_t offset) override;
    void _copy_to(Dispatcher &dispatcher, Buffer *buffer, size_t offset) override;
    void _copy_to(Dispatcher &dispatcher, Texture *texture) override;

public:
    MetalTexture(id<MTLTexture> handle, uint32_t width, uint32_t height, compute::PixelFormat format, size_t max_caches) noexcept
        : Texture{width, height, format, max_caches}, _handle{handle}, _cache{[handle device], byte_size(), max_caches} {}
    
    [[nodiscard]] id<MTLTexture> handle() const noexcept { return _handle; }
    void copy_from(Dispatcher &dispatcher, const void *data) override;
    void copy_to(Dispatcher &dispatcher, void *data) override;
    void clear_cache() override;
};

}

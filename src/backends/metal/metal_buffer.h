//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#import <compute/buffer.h>
#import "metal_host_cache.h"

namespace luisa::metal {

using compute::Buffer;
using compute::Dispatcher;

class MetalBuffer : public Buffer {

private:
    id<MTLBuffer> _handle{nullptr};
    MetalHostCache _cache;

public:
    MetalBuffer(id<MTLBuffer> buffer, size_t size, size_t host_caches) noexcept
        : Buffer{size, host_caches}, _handle{buffer}, _cache{[buffer device], size, host_caches} {}
    [[nodiscard]] id<MTLBuffer> handle() const noexcept { return _handle; }
    void upload(compute::Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) override;
    void download(compute::Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) override;
    void clear_cache() noexcept override { _cache.clear(); }
};

}

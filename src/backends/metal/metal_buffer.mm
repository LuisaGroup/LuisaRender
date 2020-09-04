//
// Created by Mike Smith on 2019/10/25.
//

#import <core/logging.h>
#import "metal_buffer.h"
#import "metal_dispatcher.h"

namespace luisa::metal {

void MetalBuffer::upload(compute::Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) {
    auto cache = _cache.obtain();
    std::memmove([cache contents], host_data, size);
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:cache sourceOffset:0 toBuffer:_handle destinationOffset:offset size:size];
    [blit_encoder endEncoding];
    dispatcher.when_completed([this, cache] { _cache.recycle(cache); });
}

void MetalBuffer::download(compute::Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) {
    auto cache = _cache.obtain();
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:_handle sourceOffset:offset toBuffer:cache destinationOffset:0 size:size];
    [blit_encoder endEncoding];
    dispatcher.when_completed([this, cache, dst = host_buffer, size] {
        std::memmove(dst, [cache contents], size);
        _cache.recycle(cache);
    });
}

void MetalBuffer::with_cache(compute::Dispatcher &dispatch, const std::function<void(void *)> &modify, size_t offset, size_t length) {
    auto cache = _cache.obtain();
    modify(reinterpret_cast<std::byte *>([cache contents]) + offset);
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatch).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:cache sourceOffset:offset toBuffer:_handle destinationOffset:offset size:length];
    [blit_encoder endEncoding];
    dispatch.when_completed([this, cache] { _cache.recycle(cache); });
}

}

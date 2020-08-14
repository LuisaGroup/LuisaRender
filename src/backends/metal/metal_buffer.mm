//
// Created by Mike Smith on 2019/10/25.
//

#import <core/logging.h>
#import "metal_buffer.h"
#import "metal_kernel.h"
#import "metal_dispatcher.h"

namespace luisa::metal {

void MetalBuffer::upload(compute::Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) {
    if (_cache == nullptr) { _cache = [[_handle device] newBufferWithLength:_size options:MTLResourceStorageModeShared]; }
    memmove([_cache contents], host_data, size);
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:_cache sourceOffset:0 toBuffer:_handle destinationOffset:offset size:size];
    [blit_encoder endEncoding];
}

void MetalBuffer::download(compute::Dispatcher &dispatcher, size_t offset, size_t size, void *host_buffer) const {
    if (_cache == nullptr) { _cache = [[_handle device] newBufferWithLength:_size options:MTLResourceStorageModeShared]; }
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:_handle sourceOffset:offset toBuffer:_cache destinationOffset:offset size:size];
    [blit_encoder endEncoding];
    dispatcher.add_callback([src = [_cache contents], dst = host_buffer, size] { std::memmove(dst, src, size); });
}

}

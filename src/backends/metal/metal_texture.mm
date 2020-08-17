//
// Created by Mike Smith on 2020/8/16.
//

#import "metal_texture.h"
#import "metal_buffer.h"
#import "metal_dispatcher.h"

namespace luisa::metal {

void MetalTexture::_copy_from(Dispatcher &dispatcher, Buffer *buffer, size_t offset) {
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:dynamic_cast<MetalBuffer *>(buffer)->handle()
                    sourceOffset:offset
               sourceBytesPerRow:pitch_byte_size()
             sourceBytesPerImage:byte_size()
                      sourceSize:MTLSizeMake(_width, _height, 1)
                       toTexture:_handle
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit_encoder endEncoding];
}

void MetalTexture::_copy_to(Dispatcher &dispatcher, Buffer *buffer, size_t offset) {
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromTexture:_handle
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(_width, _height, 1)
                         toBuffer:dynamic_cast<MetalBuffer *>(buffer)->handle()
                destinationOffset:offset
           destinationBytesPerRow:pitch_byte_size()
         destinationBytesPerImage:byte_size()];
    [blit_encoder endEncoding];
}

void MetalTexture::_copy_to(Dispatcher &dispatcher, Texture *texture) {
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromTexture:_handle toTexture:dynamic_cast<MetalTexture *>(texture)->handle()];
    [blit_encoder endEncoding];
}

void MetalTexture::copy_from(Dispatcher &dispatcher, const void *data) {
    auto cache = _cache.get();
    memmove([cache contents], data, byte_size());
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromBuffer:cache
                    sourceOffset:0
               sourceBytesPerRow:pitch_byte_size()
             sourceBytesPerImage:byte_size()
                      sourceSize:MTLSizeMake(_width, _height, 1)
                       toTexture:_handle
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit_encoder endEncoding];
    dispatcher.add_callback([this, cache] { _cache.recycle(cache); });
}

void MetalTexture::copy_to(Dispatcher &dispatcher, void *data) {
    auto cache = _cache.get();
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromTexture:_handle
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(_width, _height, 1)
                         toBuffer:cache
                destinationOffset:0
           destinationBytesPerRow:pitch_byte_size()
         destinationBytesPerImage:byte_size()];
    [blit_encoder endEncoding];
    dispatcher.add_callback([this, cache, dst = data, size = byte_size()] {
        std::memmove(dst, [cache contents], size);
        _cache.recycle(cache);
    });
}

void MetalTexture::clear_cache() {

}

}

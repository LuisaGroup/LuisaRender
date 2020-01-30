//
// Created by Mike Smith on 2019/10/27.
//

#import "metal_texture.h"
#import "metal_kernel.h"
#import "metal_buffer.h"

namespace luisa::metal {

void MetalTexture::copy_from_buffer(KernelDispatcher &dispatch, Buffer &buffer) {
    
    auto encoder = [dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer() blitCommandEncoder];
    [encoder copyFromBuffer:dynamic_cast<MetalBuffer &>(buffer).handle()
               sourceOffset:0u
          sourceBytesPerRow:bytes_per_row()
        sourceBytesPerImage:bytes_per_image()
                 sourceSize:MTLSizeMake(_size.x, _size.y, 1u)
                  toTexture:_handle
           destinationSlice:0u
           destinationLevel:0u
          destinationOrigin:MTLOriginMake(0u, 0u, 0u)];
    [encoder endEncoding];
    
}

void MetalTexture::copy_to_buffer(KernelDispatcher &dispatch, Buffer &buffer) {
    
    auto encoder = [dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer() blitCommandEncoder];
    [encoder copyFromTexture:_handle
                 sourceSlice:0u
                 sourceLevel:0u
                sourceOrigin:MTLOriginMake(0u, 0u, 0u)
                  sourceSize:MTLSizeMake(_size.x, _size.y, 1u)
                    toBuffer:dynamic_cast<MetalBuffer &>(buffer).handle()
           destinationOffset:0u
      destinationBytesPerRow:bytes_per_row()
    destinationBytesPerImage:bytes_per_image()
                     options:MTLBlitOptionNone];
    [encoder endEncoding];
    
}

}
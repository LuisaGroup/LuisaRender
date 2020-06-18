//
// Created by Mike Smith on 2019/10/25.
//

#import <core/logging.h>
#import "metal_buffer.h"
#import "metal_kernel.h"

namespace luisa::metal {

void MetalBuffer::upload(size_t offset, size_t size) {
    LUISA_EXCEPTION_IF_NOT(_storage == BufferStorage::MANAGED, "Only managed buffers can be update.");
    LUISA_EXCEPTION_IF_NOT(offset + size <= _capacity, "Buffer data overflowed");
    [_handle didModifyRange:NSMakeRange(offset, size)];
}

void MetalBuffer::synchronize(KernelDispatcher &dispatch) {
    auto command_buffer = dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder synchronizeResource:_handle];
    [blit_encoder endEncoding];
}

void *MetalBuffer::data() {
    return _handle.contents;
}

}

//
// Created by Mike Smith on 2019/10/25.
//

#import "metal_buffer.h"
#import "metal_kernel.h"

namespace luisa::metal {

void MetalBuffer::upload(const void *host_data, size_t size, size_t offset) {
    if (_storage != BufferStorage::MANAGED) { throw std::runtime_error{"only managed buffers can be update."}; }
    if (offset + size > _capacity) { throw std::runtime_error{"buffer data overflowed"}; }
    std::memmove(reinterpret_cast<uint8_t *>(_handle.contents) + offset, host_data, size);
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

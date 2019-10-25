//
// Created by Mike Smith on 2019/10/25.
//

#import "metal_buffer.h"
#import "metal_kernel.h"

void MetalBuffer::upload(const void *host_data, size_t size, size_t offset) {
    if (_storage != StorageTag::MANAGED) { throw std::runtime_error{"only managed buffers can be update."}; }
    if (offset + size > _capacity) { throw std::runtime_error{"buffer data overflowed"}; }
    std::memmove(reinterpret_cast<uint8_t *>(_buffer.contents) + offset, host_data, size);
    [_buffer didModifyRange:NSMakeRange(offset, size)];
}

void MetalBuffer::synchronize(KernelDispatcher &dispatch) {
    auto command_buffer = dynamic_cast<MetalKernelDispatcher &>(dispatch).command_buffer();
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder synchronizeResource:_buffer];
    [blit_encoder endEncoding];
}

const void *MetalBuffer::data() const {
    return _buffer.contents;
}

//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#ifndef __OBJC__
#error This file should only be used in Objective-C/C++ sources.
#endif

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <core/buffer.h>

namespace luisa::metal {

class MetalBuffer : public Buffer {

private:
    id<MTLBuffer> _handle;

public:
    MetalBuffer(id<MTLBuffer> buffer, size_t capacity, BufferStorage storage) noexcept
        : Buffer(capacity, storage), _handle{buffer} {}
    
    void upload(const void *host_data, size_t size, size_t offset) override;
    void synchronize(struct KernelDispatcher &dispatch) override;
    [[nodiscard]] void *data() override;
    [[nodiscard]] id<MTLBuffer> handle() const noexcept { return _handle; }
    
};

}

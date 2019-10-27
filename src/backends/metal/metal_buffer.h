//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

#ifndef __OBJC__
#error "This file should only be used in Objective-C/C++ sources."
#endif

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <core/buffer.h>

class MetalBuffer : public Buffer {

private:
    id<MTLBuffer> _buffer;

public:
    MetalBuffer(id<MTLBuffer> buffer, size_t capacity, BufferStorageTag storage) noexcept
        : Buffer(capacity, storage), _buffer{buffer} {}
    
    void upload(const void *host_data, size_t size, size_t offset) override;
    void synchronize(struct KernelDispatcher &dispatch) override;
    [[nodiscard]] const void *data() const override;
    [[nodiscard]] id<MTLBuffer> buffer() const noexcept { return _buffer; }
    
};



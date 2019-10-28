//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#ifndef __OBJC__
#error This file should only be used in Objective-C/C++ sources.
#endif

#import <memory>
#import <core/kernel.h>

class MetalKernelArgumentProxy : public KernelArgumentProxy {

private:
    MTLArgument *_argument;
    id<MTLComputeCommandEncoder> _encoder;

public:
    MetalKernelArgumentProxy(MTLArgument *argument, id<MTLComputeCommandEncoder> encoder) : _argument{argument}, _encoder{encoder} {}
    
    void set_buffer(Buffer &buffer, size_t offset) override;
    void set_texture(Texture &texture) override;
    void set_bytes(const void *bytes, size_t size) override;
    
};

class MetalKernelArgumentEncoder : public KernelArgumentEncoder {

private:
    MTLAutoreleasedComputePipelineReflection _info;
    id<MTLComputeCommandEncoder> _encoder;

public:
    explicit MetalKernelArgumentEncoder(MTLAutoreleasedComputePipelineReflection info, id<MTLComputeCommandEncoder> encoder) noexcept : _info{info}, _encoder{encoder} {}
    [[nodiscard]] std::unique_ptr<KernelArgumentProxy> operator[](std::string_view name) override;
    
};

class MetalKernel : public Kernel {

private:
    id<MTLComputePipelineState> _pipeline;
    MTLAutoreleasedComputePipelineReflection _reflection;

public:
    MetalKernel(id<MTLComputePipelineState> pipeline, MTLAutoreleasedComputePipelineReflection reflection) noexcept
        : _pipeline{pipeline}, _reflection{reflection} {}
    
    [[nodiscard]] auto reflection() const noexcept { return _reflection; }
    [[nodiscard]] auto pipeline() const noexcept { return _pipeline; }
    
};

class MetalKernelDispatcher : public KernelDispatcher {

private:
    id<MTLCommandBuffer> _command_buffer;

public:
    explicit MetalKernelDispatcher(id<MTLCommandBuffer> command_buffer) noexcept : _command_buffer{command_buffer} {}
    [[nodiscard]] auto command_buffer() const noexcept { return _command_buffer; }
    void operator()(Kernel &kernel, uint2 grids, uint2 grid_size, std::function<void(KernelArgumentEncoder &)> encode) override;
    
};

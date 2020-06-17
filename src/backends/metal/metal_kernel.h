//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#ifndef __OBJC__
#error This file should only be used in Objective-C/C++ sources.
#endif

#import <memory>
#import <compute/kernel.h>

namespace luisa::metal {

class MetalKernelArgumentEncoder : public KernelArgumentEncoder {

private:
    MTLComputePipelineReflection *_info;
    id<MTLComputeCommandEncoder> _encoder;
    
    [[nodiscard]] size_t _argument_index(std::string_view argument_name);

public:
    explicit MetalKernelArgumentEncoder(MTLComputePipelineReflection *info, id<MTLComputeCommandEncoder> encoder) noexcept : _info{info}, _encoder{encoder} {}
    void set_buffer(std::string_view argument_name, TypelessBuffer &buffer, size_t offset) override;
    void set_bytes(std::string_view argument_name, const void *bytes, size_t size) override;
};

class MetalKernel : public Kernel {

private:
    id<MTLFunction> _function;
    id<MTLComputePipelineState> _pipeline;
    MTLComputePipelineReflection *_reflection;

public:
    MetalKernel(id<MTLFunction> function, id<MTLComputePipelineState> pipeline, MTLComputePipelineReflection *reflection) noexcept
        : _function{function}, _pipeline{pipeline}, _reflection{reflection} {}
    
    [[nodiscard]] auto reflection() const noexcept { return _reflection; }
    [[nodiscard]] auto pipeline() const noexcept { return _pipeline; }
    [[nodiscard]] auto function() const noexcept { return _function; }
};

class MetalKernelDispatcher : public KernelDispatcher {

private:
    id<MTLCommandBuffer> _command_buffer;

public:
    explicit MetalKernelDispatcher(id<MTLCommandBuffer> command_buffer) noexcept : _command_buffer{command_buffer} {}
    [[nodiscard]] auto command_buffer() const noexcept { return _command_buffer; }
    void operator()(Kernel &kernel, uint2 grids, uint2 grid_size, std::function<void(KernelArgumentEncoder &)> encode) override;
    
};

}

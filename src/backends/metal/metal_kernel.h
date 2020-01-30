//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#ifndef __OBJC__
#error This file should only be used in Objective-C/C++ sources.
#endif

#import <memory>
#import <core/kernel.h>

namespace luisa::metal {

class MetalKernelArgumentProxy : public KernelArgumentProxy {

private:
    size_t _argument_index;
    id<MTLComputeCommandEncoder> _encoder;

public:
    MetalKernelArgumentProxy(size_t argument_index, id<MTLComputeCommandEncoder> encoder) : _argument_index{argument_index}, _encoder{encoder} {}
    
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

class MetalKernelArgumentBufferMemberProxy : public KernelArgumentBufferMemberProxy {

private:
    size_t _argument_index;
    id<MTLArgumentEncoder> _encoder;

public:
    MetalKernelArgumentBufferMemberProxy(size_t index, id<MTLArgumentEncoder> encoder) : _argument_index{index}, _encoder{encoder} {}
    void set_buffer(Buffer &argument_buffer, size_t argument_buffer_offset, Buffer &buffer, size_t offset) override;
    void set_texture(Buffer &argument_buffer, size_t argument_buffer_offset, Texture &texture) override;
    void set_bytes(Buffer &argument_buffer, size_t argument_buffer_offset, const void *bytes, size_t size) override;
};

class MetalKernelArgumentBufferEncoder : public KernelArgumentBufferEncoder {

private:
    MTLAutoreleasedArgument _info;
    id<MTLArgumentEncoder> _encoder;

public:
    MetalKernelArgumentBufferEncoder(MTLAutoreleasedArgument info, id<MTLArgumentEncoder> encoder) : _info{info}, _encoder{encoder} {}
    std::unique_ptr<KernelArgumentBufferMemberProxy> operator[](std::string_view member_name) override;
    [[nodiscard]] size_t element_size() const override { return _encoder.encodedLength; }
    [[nodiscard]] size_t element_alignment() const override { return _encoder.alignment; }
};

class MetalKernel : public Kernel {

private:
    id<MTLFunction> _function;
    id<MTLComputePipelineState> _pipeline;
    MTLAutoreleasedComputePipelineReflection _reflection;

public:
    MetalKernel(id<MTLFunction> function, id<MTLComputePipelineState> pipeline, MTLAutoreleasedComputePipelineReflection reflection) noexcept
        : _function{function}, _pipeline{pipeline}, _reflection{reflection} {}
    
    std::unique_ptr<KernelArgumentBufferEncoder> argument_buffer_encoder(std::string_view argument_name) override;
    
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

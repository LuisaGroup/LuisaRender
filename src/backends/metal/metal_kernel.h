//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#ifndef __OBJC__
#error "This file should only be used in Objective-C/C++ sources."
#endif

#import <memory>
#import <core/kernel.h>

class MetalKernelArgumentProxy : public KernelArgumentProxy {

private:
    MTLArgument *_argument;
    id<MTLCommandEncoder> _encoder;

public:
    MetalKernelArgumentProxy(MTLArgument *argument, id<MTLCommandEncoder> encoder) : _argument{argument}, _encoder{encoder} {}
    
    void set_buffer(const Buffer &buffer) override {
    
    }
    
    void set_texture(const Texture &texture) override {
    
    }
    
    void set_bytes(const void *bytes, size_t size) override {
    
    }
    
};

class MetalKernelArgumentEncoder : public KernelArgumentEncoder {

private:
    MTLAutoreleasedComputePipelineReflection _info;
    id<MTLCommandEncoder> _encoder;

public:
    explicit MetalKernelArgumentEncoder(MTLAutoreleasedComputePipelineReflection info, id<MTLCommandEncoder> encoder) noexcept : _info{info}, _encoder{encoder} {}
    
    KernelArgumentProxyWrapper operator[](std::string_view name) override {
        for (MTLArgument *argument in _info.arguments) {
            if (name == to_string(argument.name)) {
                return KernelArgumentProxyWrapper{std::make_unique<MetalKernelArgumentProxy>(argument, _encoder)};
            }
        }
        throw std::runtime_error{"argument not found in Metal compute kernel."};
    }
    
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
    
    void operator()(Kernel &kernel, uint2 grids, uint2 grid_size, std::function<void(KernelArgumentEncoder &)> encode) override {
        auto encoder = [_command_buffer computeCommandEncoder];
        [encoder dispatchThreadgroups:MTLSizeMake(grids.x, grids.y, 1) threadsPerThreadgroup:MTLSizeMake(grid_size.x, grid_size.y, 1)];
        auto &&metal_kernel = dynamic_cast<MetalKernel &>(kernel);
        [encoder setComputePipelineState:metal_kernel.pipeline()];
        MetalKernelArgumentEncoder metal_encoder{metal_kernel.reflection(), encoder};
        encode(metal_encoder);
        [encoder endEncoding];
    }
    
};

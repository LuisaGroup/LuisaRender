//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <memory>
#import <compute/kernel.h>

namespace luisa::metal {

using compute::Kernel;

class MetalKernel : public Kernel {

public:
    struct UniformBinding {
        void *dst{nullptr};
        const void *src{nullptr};
        size_t size{0ul};
    };
    
    struct ResourceUsage {
        id<MTLResource> res{nullptr};
        MTLResourceUsage usage{MTLResourceUsageRead | MTLResourceUsageWrite};
    };

private:
    id<MTLComputePipelineState> _handle;
    id<MTLBuffer> _argument_buffer;
    std::vector<UniformBinding> _argument_bindings;
    std::vector<ResourceUsage> _argument_resources;

public:
    MetalKernel(id<MTLComputePipelineState> handle, id<MTLBuffer> arg_buffer, std::vector<UniformBinding> uniforms, std::vector<ResourceUsage> res) noexcept;
    void dispatch(compute::Dispatcher &dispatcher, uint3 threadgroups, uint3 threadgroup_size) override;
    [[nodiscard]] id<MTLComputePipelineState> handle() const noexcept { return _handle; }
};

}

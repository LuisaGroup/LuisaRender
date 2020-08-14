//
// Created by Mike Smith on 2019/10/24.
//

#import "metal_kernel.h"
#import "metal_dispatcher.h"

namespace luisa::metal {

void MetalKernel::dispatch(Dispatcher &dispatcher, uint3 threadgroups, uint3 threadgroup_size) {
    for (auto[dst, src, size] : _argument_bindings) { std::memmove(dst, src, size); }
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto command_encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [command_encoder setComputePipelineState:_handle];
    [command_encoder setBuffer:_argument_buffer offset:0u atIndex:0u];
    for (auto[res, usage] : _argument_resources) { [command_encoder useResource:res usage:usage]; }
    [command_encoder dispatchThreadgroups:MTLSizeMake(threadgroups.x, threadgroups.y, threadgroups.z)
                    threadsPerThreadgroup:MTLSizeMake(threadgroup_size.x, threadgroup_size.y, threadgroup_size.z)];
    [command_encoder endEncoding];
}

MetalKernel::MetalKernel(id<MTLComputePipelineState> handle, id<MTLBuffer> arg_buffer, std::vector<UniformBinding> uniforms, std::vector<ResourceUsage> res) noexcept
    : _handle{handle}, _argument_buffer{arg_buffer}, _argument_bindings{std::move(uniforms)}, _argument_resources{std::move(res)} {}
    
}

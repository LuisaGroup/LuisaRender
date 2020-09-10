//
// Created by Mike Smith on 2019/10/24.
//

#import <core/logging.h>

#import "metal_dispatcher.h"
#import "metal_kernel.h"
#import "metal_buffer.h"
#import "metal_texture.h"

namespace luisa::metal {

void MetalKernel::_dispatch(Dispatcher &dispatcher, uint2 threadgroups, uint2 threadgroup_size) {
    
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto command_encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [command_encoder setComputePipelineState:_handle];
    
    // encode arguments
    uint buffer_id = 0;
    uint texture_id = 0;
    for (auto &&r : _resources) {
        if (r.buffer != nullptr) {  // buffer
            auto mtl_buffer = dynamic_cast<MetalBuffer *>(r.buffer.get())->handle();
            [command_encoder setBuffer:mtl_buffer offset:0u atIndex:buffer_id];
            buffer_id++;
        } else if (r.texture != nullptr) {  // texture
            auto mtl_texture = dynamic_cast<MetalTexture *>(r.texture.get())->handle();
            [command_encoder setTexture:mtl_texture atIndex:texture_id];
            texture_id++;
        } else {
            LUISA_ERROR("Invalid argument.");
        }
    }
    if (!_uniforms.empty()) {
        if (_uniform_buffer.empty()) {
            _uniform_buffer.resize(_uniforms.back().offset + std::max(_uniforms.back().immutable.size(), _uniforms.back().binding_size));
            for (auto &&u : _uniforms) {
                if (u.binding == nullptr) { std::memmove(_uniform_buffer.data() + u.offset, u.immutable.data(), u.immutable.size()); }
            }
        }
        for (auto &&u : _uniforms) {
            if (u.binding != nullptr) { std::memmove(_uniform_buffer.data() + u.offset, u.binding, u.binding_size); }
        }
        [command_encoder setBytes:_uniform_buffer.data() length:_uniform_buffer.size() atIndex:buffer_id];
    }
    [command_encoder dispatchThreadgroups:MTLSizeMake(threadgroups.x, threadgroups.y, 1u)
                    threadsPerThreadgroup:MTLSizeMake(threadgroup_size.x, threadgroup_size.y, 1u)];
    [command_encoder endEncoding];
}

MetalKernel::MetalKernel(id<MTLComputePipelineState> handle, std::vector<Kernel::Resource> res, std::vector<Kernel::Uniform> uniforms) noexcept
    : Kernel{std::move(res), std::move(uniforms)}, _handle{handle} {}

}

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
        std::vector<std::byte> bytes(_uniforms.back().offset + std::max(_uniforms.back().immutable.size(), _uniforms.back().binding_size));
        for (auto &&u : _uniforms) {
            if (u.binding != nullptr) {
                std::memmove(bytes.data() + u.offset, u.binding, u.binding_size);
            } else {
                std::memmove(bytes.data() + u.offset, u.immutable.data(), u.immutable.size());
            }
        }
        [command_encoder setBytes:bytes.data() length:bytes.size() atIndex:buffer_id];
    }
    [command_encoder dispatchThreadgroups:MTLSizeMake(threadgroups.x, threadgroups.y, 1u)
                    threadsPerThreadgroup:MTLSizeMake(threadgroup_size.x, threadgroup_size.y, 1u)];
    [command_encoder endEncoding];
}

MetalKernel::MetalKernel(id<MTLComputePipelineState> handle, std::vector<Resource> res, std::vector<Uniform> uniforms) noexcept
    : _handle{handle}, _resources{std::move(res)}, _uniforms{uniforms} {}

}

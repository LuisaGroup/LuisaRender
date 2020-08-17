//
// Created by Mike Smith on 2019/10/24.
//

#import <core/logging.h>

#import "metal_kernel.h"
#import "metal_dispatcher.h"

namespace luisa::metal {

void MetalKernel::_dispatch(Dispatcher &dispatcher, uint3 threadgroups, uint3 threadgroup_size) {
    
    id<MTLBuffer> argument_buffer = nullptr;
    
    if (_argument_bindings.empty()) {
        argument_buffer = _available_argument_buffers.front();
    } else {
        argument_buffer = _get_argument_buffer();
        [_argument_encoder setArgumentBuffer:argument_buffer offset:0];
        for (auto[index, size, src] : _argument_bindings) {
            std::memmove([_argument_encoder constantDataAtIndex:index], src, size);
        }
        dispatcher.add_callback([this, argument_buffer] {
            std::lock_guard lock{_argument_buffer_mutex};
            _available_argument_buffers.emplace_back(argument_buffer);
        });
    }
    
    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto command_encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [command_encoder setComputePipelineState:_handle];
    [command_encoder setBuffer:argument_buffer offset:0u atIndex:0u];
    for (auto argument : _arguments) {
        std::visit([&](auto &&arg) noexcept {
            using Type = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<Type, BufferArgument> || std::is_same_v<Type, TextureArgument>) {
                [command_encoder useResource:arg.handle usage:arg.usage];
            }
        }, argument);
    }
    [command_encoder dispatchThreadgroups:MTLSizeMake(threadgroups.x, threadgroups.y, threadgroups.z)
                    threadsPerThreadgroup:MTLSizeMake(threadgroup_size.x, threadgroup_size.y, threadgroup_size.z)];
    [command_encoder endEncoding];
}

MetalKernel::MetalKernel(id<MTLComputePipelineState> handle,
                         std::vector<Uniform> uniforms,
                         std::vector<Argument> args,
                         id<MTLArgumentEncoder> arg_enc) noexcept
    : _handle{handle}, _argument_bindings{std::move(uniforms)}, _arguments{std::move(args)}, _argument_encoder{arg_enc} {
    
    if (_argument_bindings.empty()) {
        _available_argument_buffers.emplace_back(_create_argument_buffer());
    }
}

id<MTLBuffer> MetalKernel::_create_argument_buffer() {
    auto device = [_handle device];
    auto buffer = [device newBufferWithLength:_argument_encoder.encodedLength options:MTLResourceOptionCPUCacheModeWriteCombined];
    [_argument_encoder setArgumentBuffer:buffer offset:0];
    for (auto &&argument : _arguments) {
        std::visit([&](auto &&arg) noexcept {
            using Type = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<Type, ImmutableArgument>) {
                std::memmove([_argument_encoder constantDataAtIndex:arg.index], arg.data.data(), arg.data.size());
            } else if constexpr (std::is_same_v<Type, BufferArgument>) {
                [_argument_encoder setBuffer:arg.handle offset:arg.offset atIndex:arg.index];
            } else if constexpr (std::is_same_v<Type, TextureArgument>) {
                [_argument_encoder setTexture:arg.handle atIndex:arg.index];
            }
        }, argument);
    }
    LUISA_INFO("Created argument buffer #", _argument_buffer_count++, " with length ", _argument_encoder.encodedLength, " for kernel launch.");
    return buffer;
}

id<MTLBuffer> MetalKernel::_get_argument_buffer() {
    std::lock_guard lock{_argument_buffer_mutex};
    if (_available_argument_buffers.empty()) {
        return _create_argument_buffer();
    }
    auto buffer = _available_argument_buffers.back();
    _available_argument_buffers.pop_back();
    return buffer;
}

}

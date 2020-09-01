//
// Created by Mike Smith on 2019/10/24.
//

#import <core/logging.h>

#import "metal_dispatcher.h"
#import "metal_kernel.h"

namespace luisa::metal {

void MetalKernel::_dispatch(Dispatcher &dispatcher, uint2 threadgroups, uint2 threadgroup_size) {

    auto argument_buffer = _get_argument_buffer();
    if (!_argument_bindings.empty()) {
        [_argument_encoder setArgumentBuffer:argument_buffer.handle offset:argument_buffer.offset];
        for (auto [index, size, src] : _argument_bindings) {
            std::memmove([_argument_encoder constantDataAtIndex:index], src, size);
        }
        dispatcher.when_completed([this, argument_buffer] { _argument_buffer_pool.recycle(argument_buffer); });
    }

    auto command_buffer = dynamic_cast<MetalDispatcher &>(dispatcher).handle();
    auto command_encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [command_encoder setComputePipelineState:_handle];
    [command_encoder setBuffer:argument_buffer.handle offset:argument_buffer.offset atIndex:0u];
    for (auto argument : _arguments) {
        std::visit([&](auto &&arg) noexcept {
            using Type = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<Type, BufferArgument> || std::is_same_v<Type, TextureArgument>) {
                [command_encoder useResource:arg.handle usage:arg.usage];
            }
        }, argument);
    }
    [command_encoder dispatchThreadgroups:MTLSizeMake(threadgroups.x, threadgroups.y, 1u)
                    threadsPerThreadgroup:MTLSizeMake(threadgroup_size.x, threadgroup_size.y, 1u)];
    [command_encoder endEncoding];
}

MetalKernel::MetalKernel(id<MTLComputePipelineState> handle,
                         std::vector<Uniform> uniforms,
                         std::vector<Argument> args,
                         id<MTLArgumentEncoder> arg_enc) noexcept
    : _handle{handle},
      _argument_bindings{std::move(uniforms)},
      _arguments{std::move(args)},
      _argument_encoder{arg_enc},
      _argument_buffer_pool{[handle device], arg_enc.encodedLength, arg_enc.alignment} {}

ArgumentBufferView MetalKernel::_get_argument_buffer() {
    if (_argument_bindings.empty()) {
        if (_constant_argument_buffer == nullptr) {
            auto device = [_handle device];
            _constant_argument_buffer = [device newBufferWithLength:_argument_encoder.encodedLength
                                                            options:MTLCPUCacheModeWriteCombined | MTLHazardTrackingModeUntracked];
            _initialize_argument_buffer(_constant_argument_buffer, 0u);
        }
        return {_constant_argument_buffer, 0u, true};
    }
    auto buffer = _argument_buffer_pool.obtain();
    if (!buffer.initialized) { _initialize_argument_buffer(buffer.handle, buffer.offset); }
    return {buffer.handle, buffer.offset, true};
}

void MetalKernel::_initialize_argument_buffer(id<MTLBuffer> buffer, size_t offset) {
    [_argument_encoder setArgumentBuffer:buffer offset:offset];
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
}

}

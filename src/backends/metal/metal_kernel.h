//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#import <memory>
#import <variant>
#import <type_traits>
#import <condition_variable>

#import <Metal/Metal.h>
#import <compute/kernel.h>

namespace luisa::metal {

using compute::Kernel;

class MetalKernel : public Kernel {

public:
    struct Uniform {
        uint32_t index{0u};
        uint32_t size{0u};
        const void *src{nullptr};
    };
    
    struct ImmutableArgument {
        std::vector<std::byte> data;
        uint32_t index{0u};
    };
    
    struct BufferArgument {
        id<MTLBuffer> handle{nullptr};
        MTLResourceUsage usage{};
        uint32_t offset{0u};
        uint32_t index{0u};
    };
    
    struct TextureArgument {
        id<MTLTexture> handle{nullptr};
        MTLResourceUsage usage{};
        uint32_t index{0u};
    };
    
    using Argument = std::variant<ImmutableArgument, BufferArgument, TextureArgument>;

private:
    id<MTLComputePipelineState> _handle;
    std::vector<Uniform> _argument_bindings;
    std::vector<Argument> _arguments;
    id<MTLArgumentEncoder> _argument_encoder;
    std::vector<id<MTLBuffer>> _available_argument_buffers;
    std::mutex _argument_buffer_mutex;
    uint32_t _argument_buffer_count{0u};
    
    [[nodiscard]] id<MTLBuffer> _create_argument_buffer();
    [[nodiscard]] id<MTLBuffer> _get_argument_buffer();
    
protected:
    void _dispatch(compute::Dispatcher &dispatcher, uint3 threadgroups, uint3 threadgroup_size) override;

public:
    MetalKernel(
        id<MTLComputePipelineState> handle,
        std::vector<Uniform> uniforms,
        std::vector<Argument> args,
        id<MTLArgumentEncoder> arg_enc) noexcept;
    [[nodiscard]] id<MTLComputePipelineState> handle() const noexcept { return _handle; }
};

}

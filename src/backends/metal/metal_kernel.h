//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#import <algorithm>
#import <memory>
#import <mutex>
#import <type_traits>
#import <variant>

#import <Metal/Metal.h>

#import <core/logging.h>
#import <core/platform.h>
#import <compute/kernel.h>
#import <compute/buffer.h>
#import <compute/texture.h>

namespace luisa::metal {

using compute::Kernel;
using compute::Buffer;
using compute::Texture;

class MetalKernel : public Kernel {

public:
    struct Resource {
        std::shared_ptr<Buffer> buffer{nullptr};
        std::shared_ptr<Texture> texture{nullptr};
    };
    
    struct Uniform {
        std::vector<std::byte> immutable;
        const void *binding{nullptr};
        size_t binding_size{};
        size_t offset{0u};
    };

private:
    id<MTLComputePipelineState> _handle;
    std::vector<Resource> _resources;
    std::vector<Uniform> _uniforms;

protected:
    void _dispatch(compute::Dispatcher &dispatcher, uint2 threadgroups, uint2 threadgroup_size) override;

public:
    MetalKernel(id<MTLComputePipelineState> handle, std::vector<Resource> res, std::vector<Uniform> uniforms) noexcept;
    [[nodiscard]] id<MTLComputePipelineState> handle() const noexcept { return _handle; }
};

}

//
// Created by Mike Smith on 2019/10/24.
//

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <util/string_manipulation.h>
#import "metal_kernel.h"
#import "metal_buffer.h"

void MetalKernelDispatcher::operator()(Kernel &kernel, uint2 grids, uint2 grid_size, std::function<void(KernelArgumentEncoder &)> encode) {
    auto encoder = [_command_buffer computeCommandEncoder];
    [encoder dispatchThreadgroups:MTLSizeMake(grids.x, grids.y, 1) threadsPerThreadgroup:MTLSizeMake(grid_size.x, grid_size.y, 1)];
    auto &&metal_kernel = dynamic_cast<MetalKernel &>(kernel);
    [encoder setComputePipelineState:metal_kernel.pipeline()];
    MetalKernelArgumentEncoder metal_encoder{metal_kernel.reflection(), encoder};
    encode(metal_encoder);
    [encoder endEncoding];
}

KernelArgumentProxyWrapper MetalKernelArgumentEncoder::operator[](std::string_view name) {
    for (MTLArgument *argument in _info.arguments) {
        if (name == to_string(argument.name)) {
            return KernelArgumentProxyWrapper{std::make_unique<MetalKernelArgumentProxy>(argument, _encoder)};
        }
    }
    throw std::runtime_error{"argument not found in Metal compute kernel."};
}

void MetalKernelArgumentProxy::set_buffer(Buffer &buffer) {
    [_encoder setBuffer:dynamic_cast<const MetalBuffer &>(buffer).handle() offset:0 atIndex:_argument.index];
}

void MetalKernelArgumentProxy::set_texture(Texture &texture) {

}

void MetalKernelArgumentProxy::set_bytes(const void *bytes, size_t size) {
    [_encoder setBytes:bytes length:size atIndex:_argument.index];
}

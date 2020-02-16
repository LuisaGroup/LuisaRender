//
// Created by Mike Smith on 2019/10/24.
//

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <util/string_manipulation.h>
#import "metal_kernel.h"
#import "metal_buffer.h"

namespace luisa::metal {

void MetalKernelDispatcher::operator()(Kernel &kernel, uint2 grids, uint2 grid_size, std::function<void(KernelArgumentEncoder &)> encode) {
    auto encoder = [_command_buffer computeCommandEncoder];
    auto &&metal_kernel = dynamic_cast<MetalKernel &>(kernel);
    MetalKernelArgumentEncoder metal_encoder{metal_kernel.reflection(), encoder};
    encode(metal_encoder);
    [encoder setComputePipelineState:metal_kernel.pipeline()];
    [encoder dispatchThreadgroups:MTLSizeMake(grids.x, grids.y, 1) threadsPerThreadgroup:MTLSizeMake(grid_size.x, grid_size.y, 1)];
    [encoder endEncoding];
}

void MetalKernelArgumentEncoder::set_buffer(std::string_view argument_name, TypelessBuffer &buffer, size_t offset) {
    [_encoder setBuffer:dynamic_cast<MetalBuffer &>(buffer).handle() offset:offset atIndex:_argument_index(argument_name)];
}

void MetalKernelArgumentEncoder::set_bytes(std::string_view argument_name, const void *bytes, size_t size) {
    [_encoder setBytes:bytes length:size atIndex:_argument_index(argument_name)];
}

size_t MetalKernelArgumentEncoder::_argument_index(std::string_view argument_name) {
    for (MTLArgument *argument in _info.arguments) {
        if (argument_name == to_string(argument.name)) { return argument.index; }
    }
    throw std::runtime_error{"argument not found in Metal compute kernel."};
}

}

//
// Created by Mike Smith on 2019/10/24.
//

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <util/string_manipulation.h>
#import "metal_kernel.h"
#import "metal_buffer.h"
#import "metal_texture.h"

void MetalKernelDispatcher::operator()(Kernel &kernel, uint2 grids, uint2 grid_size, std::function<void(KernelArgumentEncoder &)> encode) {
    auto encoder = [_command_buffer computeCommandEncoder];
    auto &&metal_kernel = dynamic_cast<MetalKernel &>(kernel);
    MetalKernelArgumentEncoder metal_encoder{metal_kernel.reflection(), encoder};
    encode(metal_encoder);
    [encoder setComputePipelineState:metal_kernel.pipeline()];
    [encoder dispatchThreadgroups:MTLSizeMake(grids.x, grids.y, 1) threadsPerThreadgroup:MTLSizeMake(grid_size.x, grid_size.y, 1)];
    [encoder endEncoding];
}

std::unique_ptr<KernelArgumentProxy> MetalKernelArgumentEncoder::operator[](std::string_view name) {
    for (MTLArgument *argument in _info.arguments) {
        if (name == to_string(argument.name)) {
            return std::make_unique<MetalKernelArgumentProxy>(argument.index, _encoder);
        }
    }
    throw std::runtime_error{"argument not found in Metal compute kernel."};
}

void MetalKernelArgumentProxy::set_buffer(Buffer &buffer, size_t offset) {
    [_encoder setBuffer:dynamic_cast<MetalBuffer &>(buffer).handle() offset:offset atIndex:_argument_index];
}

void MetalKernelArgumentProxy::set_texture(Texture &texture) {
    [_encoder setTexture:dynamic_cast<MetalTexture &>(texture).handle() atIndex:_argument_index];
}

void MetalKernelArgumentProxy::set_bytes(const void *bytes, size_t size) {
    [_encoder setBytes:bytes length:size atIndex:_argument_index];
}

std::unique_ptr<KernelArgumentBufferMemberProxy> MetalKernelArgumentBufferEncoder::operator[](std::string_view member_name) {
    for (MTLStructMember *member in _info.bufferPointerType.elementStructType.members) {
        if (member_name == to_string(member.name)) {
            if (member.dataType == MTLDataTypePointer || member.dataType == MTLDataTypeTexture || member.dataType == MTLDataTypeSampler) {
                return std::make_unique<MetalKernelArgumentBufferMemberProxy>(member.argumentIndex, _encoder);
            } else {
            
            }
        }
    }
    throw std::runtime_error{"argument not found in Metal compute kernel."};
}

std::unique_ptr<KernelArgumentBufferEncoder> MetalKernel::argument_buffer_encoder(std::string_view argument_name) {
    for (MTLArgument *argument in _reflection.arguments) {
        if (argument_name == to_string(argument.name)) {
            MTLAutoreleasedArgument reflection;
            auto encoder = [_function newArgumentEncoderWithBufferIndex:argument.index reflection:&reflection];
            [encoder autorelease];
            return std::make_unique<MetalKernelArgumentBufferEncoder>(reflection, encoder);
        }
    }
    return std::unique_ptr<KernelArgumentBufferEncoder>();
}

void MetalKernelArgumentBufferMemberProxy::set_buffer(Buffer &argument_buffer, size_t argument_buffer_offset, Buffer &buffer, size_t offset) {
    [_encoder setArgumentBuffer:dynamic_cast<MetalBuffer &>(argument_buffer).handle() offset:argument_buffer_offset];
    [_encoder setBuffer:dynamic_cast<MetalBuffer &>(buffer).handle() offset:offset atIndex:_argument_index];
}

void MetalKernelArgumentBufferMemberProxy::set_texture(Buffer &argument_buffer, size_t argument_buffer_offset, Texture &texture) {
    [_encoder setArgumentBuffer:dynamic_cast<MetalBuffer &>(argument_buffer).handle() offset:argument_buffer_offset];
    [_encoder setTexture:dynamic_cast<MetalTexture &>(texture).handle() atIndex:_argument_index];
}

void MetalKernelArgumentBufferMemberProxy::set_bytes(Buffer &argument_buffer, size_t argument_buffer_offset, const void *bytes, size_t size) {
    [_encoder setArgumentBuffer:dynamic_cast<MetalBuffer &>(argument_buffer).handle() offset:argument_buffer_offset];
    auto p = [_encoder constantDataAtIndex:_argument_index];
    std::memmove(p, bytes, size);
}

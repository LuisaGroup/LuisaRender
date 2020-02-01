//
// Created by Mike Smith on 2019/10/24.
//

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <core/ray.h>

#import <util/resource_manager.h>
#import <util/string_manipulation.h>

#import "metal_device.h"
#import "metal_buffer.h"
#import "metal_kernel.h"
#import "metal_acceleration.h"

namespace luisa::metal {

struct MetalDeviceWrapper { id<MTLDevice> device; };
struct MetalLibraryWrapper { id<MTLLibrary> library; };
struct MetalCommandQueueWrapper { id<MTLCommandQueue> queue; };

MetalDevice::MetalDevice()
    : _device_wrapper{std::make_unique<MetalDeviceWrapper>()},
      _library_wrapper{std::make_unique<MetalLibraryWrapper>()},
      _command_queue_wrapper{std::make_unique<MetalCommandQueueWrapper>()} {
    
    _device_wrapper->device = MTLCreateSystemDefaultDevice();
    _command_queue_wrapper->queue = [_device_wrapper->device newCommandQueue];
    
    auto library_path = make_objc_string(ResourceManager::instance().working_path("kernels/bin/kernels.metallib").c_str());
    _library_wrapper->library = [_device_wrapper->device newLibraryWithFile:library_path error:nullptr];
    
}

std::unique_ptr<Kernel> MetalDevice::create_kernel(std::string_view function_name) {
    
    auto descriptor = [[MTLComputePipelineDescriptor alloc] init];
    [descriptor autorelease];
    
    auto function = [_library_wrapper->library newFunctionWithName:make_objc_string(function_name)];
    [function autorelease];
    
    descriptor.computeFunction = function;
    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
    
    MTLAutoreleasedComputePipelineReflection reflection;
    auto pipeline = [_device_wrapper->device newComputePipelineStateWithDescriptor:descriptor
                                                                           options:MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo
                                                                        reflection:&reflection
                                                                             error:nullptr];
    [pipeline autorelease];
    
    std::cout << "Function: " << function_name << std::endl;
    for (MTLArgument *argument in reflection.arguments) {
        if (argument.type == MTLArgumentTypeBuffer) {
            std::cout << "  [" << argument.index << "] " << to_string(argument.name) << ": " << argument.bufferDataSize << "(" << argument.bufferAlignment << ")";
            if (argument.bufferPointerType.elementIsArgumentBuffer) {
                auto encoder = [function newArgumentEncoderWithBufferIndex:argument.index];
                [encoder autorelease];
                std::cout << " " << encoder.encodedLength;
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    
    return std::make_unique<MetalKernel>(function, pipeline, reflection);
    
}

void MetalDevice::launch(std::function<void(KernelDispatcher &)> dispatch) {
    auto command_buffer = [_command_queue_wrapper->queue commandBuffer];
    MetalKernelDispatcher dispatcher{command_buffer};
    dispatch(dispatcher);
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

void MetalDevice::launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) {
    auto command_buffer = [_command_queue_wrapper->queue commandBuffer];
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) { callback(); }];
    MetalKernelDispatcher dispatcher{command_buffer};
    dispatch(dispatcher);
    [command_buffer commit];
}

std::unique_ptr<Acceleration> MetalDevice::create_acceleration(Buffer &position_buffer, size_t stride, size_t triangle_count) {
    
    auto accelerator = [[MPSTriangleAccelerationStructure alloc] initWithDevice:_device_wrapper->device];
    [accelerator autorelease];
    accelerator.vertexBuffer = dynamic_cast<MetalBuffer &>(position_buffer).handle();
    accelerator.vertexStride = stride;
    accelerator.triangleCount = triangle_count;
    [accelerator rebuild];
    
    auto ray_intersector = [[MPSRayIntersector alloc] initWithDevice:_device_wrapper->device];
    [ray_intersector autorelease];
    ray_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    ray_intersector.intersectionDataType = MPSIntersectionDataTypeDistancePrimitiveIndexCoordinates;
    ray_intersector.rayStride = sizeof(Ray);
    
    auto shadow_ray_intersector = [[MPSRayIntersector alloc] initWithDevice:_device_wrapper->device];
    [shadow_ray_intersector autorelease];
    shadow_ray_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    shadow_ray_intersector.intersectionDataType = MPSIntersectionDataTypeDistance;
    shadow_ray_intersector.rayStride = sizeof(Ray);
    
    return std::make_unique<MetalAcceleration>(accelerator, ray_intersector, shadow_ray_intersector);
}

std::unique_ptr<Buffer> MetalDevice::allocate_buffer(size_t capacity, BufferStorage storage) {
    
    auto buffer = [_device_wrapper->device newBufferWithLength:capacity
                                                       options:storage == BufferStorage::DEVICE_PRIVATE ? MTLResourceStorageModePrivate : MTLResourceStorageModeManaged];
    [buffer autorelease];
    return std::make_unique<MetalBuffer>(buffer, capacity, storage);
}

}
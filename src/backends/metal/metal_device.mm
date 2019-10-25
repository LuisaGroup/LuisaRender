//
// Created by Mike Smith on 2019/10/24.
//

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <util/resource_manager.h>
#import <util/string_manipulation.h>

#import "metal_device.h"
#import "metal_kernel.h"

struct MetalDeviceWrapper { id<MTLDevice> device; };
struct MetalLibraryWrapper { id<MTLLibrary> library; };
struct MetalCommandQueueWrapper { id<MTLCommandQueue> queue; };

MetalDevice::MetalDevice()
    : _device_wrapper{std::make_unique<MetalDeviceWrapper>()},
      _library_wrapper{std::make_unique<MetalLibraryWrapper>()},
      _command_queue_wrapper{std::make_unique<MetalCommandQueueWrapper>()} {
    
    _device_wrapper->device = MTLCreateSystemDefaultDevice();
    _command_queue_wrapper->queue = [_device_wrapper->device newCommandQueue];
    
    auto library_path = make_objc_string(ResourceManager::instance().binary_path("kernels.metallib").c_str());
    _library_wrapper->library = [_device_wrapper->device newLibraryWithFile:library_path error:nullptr];
    
}

std::shared_ptr<Kernel> MetalDevice::create_kernel(std::string_view function_name) {
    
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
    return std::make_unique<MetalKernel>(pipeline, reflection);
    
}

std::shared_ptr<Texture> MetalDevice::create_texture() {
    return std::shared_ptr<Texture>();
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

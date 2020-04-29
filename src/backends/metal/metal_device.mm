//
// Created by Mike Smith on 2019/10/24.
//

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <core/geometry.h>

#import <util/resource_manager.h>
#import <util/string_manipulation.h>

#import "metal_device.h"
#import "metal_buffer.h"
#import "metal_kernel.h"
#import "metal_acceleration.h"

namespace luisa::metal {

struct MetalDeviceWrapper {
    id<MTLDevice> device;
    ~MetalDeviceWrapper() noexcept = default;
};

struct MetalLibraryWrapper {
    id<MTLLibrary> library;
    ~MetalLibraryWrapper() noexcept = default;
};

struct MetalCommandQueueWrapper {
    id<MTLCommandQueue> queue;
    ~MetalCommandQueueWrapper() noexcept = default;
};

struct MetalFunctionWrapper {
    
    id<MTLFunction> function;
    id<MTLComputePipelineState> pipeline;
    MTLComputePipelineReflection *reflection;
    
    MetalFunctionWrapper(id<MTLFunction> f, id<MTLComputePipelineState> p, MTLComputePipelineReflection *r) noexcept
        : function{f}, pipeline{p}, reflection{r} {}
};

MetalDevice::MetalDevice()
    : _device_wrapper{std::make_unique<MetalDeviceWrapper>()},
      _library_wrapper{std::make_unique<MetalLibraryWrapper>()},
      _command_queue_wrapper{std::make_unique<MetalCommandQueueWrapper>()} {
    
    _device_wrapper->device = MTLCreateSystemDefaultDevice();
    _command_queue_wrapper->queue = [_device_wrapper->device newCommandQueue];
    
    auto library_path = make_objc_string(ResourceManager::instance().working_path("kernels/metal/bin/kernels.metallib").c_str());
    _library_wrapper->library = [_device_wrapper->device newLibraryWithFile:library_path error:nullptr];
    
}

std::unique_ptr<Kernel> MetalDevice::create_kernel(std::string_view function_name) {
    
    std::string name{function_name};
    auto iter = _function_wrappers.find(name);
    
    if (iter == _function_wrappers.end()) {
        auto descriptor = [[MTLComputePipelineDescriptor alloc] init];
        auto function = [_library_wrapper->library newFunctionWithName:make_objc_string(name)];
        
        descriptor.computeFunction = function;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
        
        MTLAutoreleasedComputePipelineReflection reflection;
        auto pipeline = [_device_wrapper->device newComputePipelineStateWithDescriptor:descriptor
                                                                               options:MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo
                                                                            reflection:&reflection
                                                                                 error:nullptr];
        
        iter = _function_wrappers.emplace(std::move(name), std::make_unique<MetalFunctionWrapper>(function, pipeline, reflection)).first;
    }
    
    auto function_wrapper = iter->second.get();
    return std::make_unique<MetalKernel>(function_wrapper->function, function_wrapper->pipeline, function_wrapper->reflection);
}

void MetalDevice::launch(std::function<void(KernelDispatcher &)> dispatch) {
    auto command_buffer = [_command_queue_wrapper->queue commandBuffer];
    MetalKernelDispatcher dispatcher{command_buffer};
    dispatch(dispatcher);
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

void MetalDevice::_launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) {
    auto command_buffer = [_command_queue_wrapper->queue commandBuffer];
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) { callback(); }];
    MetalKernelDispatcher dispatcher{command_buffer};
    dispatch(dispatcher);
    [command_buffer commit];
}

std::unique_ptr<TypelessBuffer> MetalDevice::allocate_buffer(size_t capacity, BufferStorage storage) {
    
    auto buffer = [_device_wrapper->device newBufferWithLength:capacity
                                                       options:storage == BufferStorage::DEVICE_PRIVATE ? MTLResourceStorageModePrivate : MTLResourceStorageModeManaged];
    return std::make_unique<MetalBuffer>(buffer, capacity, storage);
}

std::unique_ptr<Acceleration> MetalDevice::create_acceleration(Geometry &geometry) {
    
    auto acceleration_group = [[MPSAccelerationStructureGroup alloc] initWithDevice:_device_wrapper->device];
    auto instance_acceleration = [[MPSInstanceAccelerationStructure alloc] initWithGroup:acceleration_group];
    
    auto acceleration_structures = [NSMutableArray array];
    instance_acceleration.accelerationStructures = acceleration_structures;
    for (auto &&entity : geometry.entities()) {
        auto triangle_acceleration = [[MPSTriangleAccelerationStructure alloc] initWithGroup:acceleration_group];
        triangle_acceleration.vertexBuffer = dynamic_cast<MetalBuffer &>(entity->position_buffer().typeless_buffer()).handle();
        triangle_acceleration.vertexBufferOffset = entity->position_buffer().byte_offset();
        triangle_acceleration.vertexStride = sizeof(float4);
        triangle_acceleration.indexBuffer = dynamic_cast<MetalBuffer &>(entity->index_buffer().typeless_buffer()).handle();
        triangle_acceleration.indexBufferOffset = entity->index_buffer().byte_offset();
        triangle_acceleration.indexType = MPSDataTypeUInt32;
        triangle_acceleration.triangleCount = entity->triangle_count();
        [triangle_acceleration rebuild];
        [acceleration_structures addObject:triangle_acceleration];
    }
    
    instance_acceleration.instanceBuffer = dynamic_cast<MetalBuffer &>(geometry.entity_index_buffer().typeless_buffer()).handle();
    instance_acceleration.instanceBufferOffset = geometry.entity_index_buffer().byte_offset();
    instance_acceleration.instanceCount = geometry.entity_index_buffer().size();
    instance_acceleration.transformBuffer = dynamic_cast<MetalBuffer &>(geometry.transform_buffer().typeless_buffer()).handle();
    instance_acceleration.transformBufferOffset = geometry.transform_buffer().byte_offset();
    
    // Note: metal provides optimization for static scenes without instances and dynamic transforms
    instance_acceleration.transformType = geometry.static_instances().empty() && geometry.dynamic_shapes().empty() && geometry.dynamic_instances().empty() ?
                                          MPSTransformTypeIdentity : MPSTransformTypeFloat4x4;
    instance_acceleration.usage =
        geometry.dynamic_shapes().empty() && geometry.dynamic_instances().empty() ? MPSAccelerationStructureUsageNone : MPSAccelerationStructureUsageRefit;
    
    [instance_acceleration rebuild];
    
    auto ray_intersector = [[MPSRayIntersector alloc] initWithDevice:_device_wrapper->device];
    ray_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    ray_intersector.rayStride = sizeof(Ray);
    ray_intersector.intersectionDataType = MPSIntersectionDataTypeDistancePrimitiveIndexInstanceIndexCoordinates;
    ray_intersector.intersectionStride = sizeof(ClosestHit);
    ray_intersector.boundingBoxIntersectionTestType = MPSBoundingBoxIntersectionTestTypeAxisAligned;
    ray_intersector.triangleIntersectionTestType = MPSTriangleIntersectionTestTypeWatertight;
    
    auto shadow_ray_intersector = [[MPSRayIntersector alloc] initWithDevice:_device_wrapper->device];
    shadow_ray_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    shadow_ray_intersector.rayStride = sizeof(Ray);
    shadow_ray_intersector.intersectionDataType = MPSIntersectionDataTypeDistance;
    shadow_ray_intersector.intersectionStride = sizeof(AnyHit);
    shadow_ray_intersector.boundingBoxIntersectionTestType = MPSBoundingBoxIntersectionTestTypeAxisAligned;
    shadow_ray_intersector.triangleIntersectionTestType = MPSTriangleIntersectionTestTypeWatertight;
    
    return std::make_unique<MetalAcceleration>(acceleration_group, instance_acceleration, ray_intersector, shadow_ray_intersector);
}

}

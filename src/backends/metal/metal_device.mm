//
// Created by Mike Smith on 2019/10/24.
//

#import <cstdlib>
#import <string>
#import <fstream>
#import <string_view>

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
    explicit MetalLibraryWrapper(id<MTLLibrary> l) noexcept: library{l} {}
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
      _command_queue_wrapper{std::make_unique<MetalCommandQueueWrapper>()} {
    
    _device_wrapper->device = MTLCreateSystemDefaultDevice();
    _command_queue_wrapper->queue = [_device_wrapper->device newCommandQueue];
}

std::unique_ptr<Kernel> MetalDevice::load_kernel(std::string_view function_name) {
    
    auto separator_pos = function_name.rfind("::");
    LUISA_EXCEPTION_IF(separator_pos == std::string_view::npos, "Expected separator \"::\" in function name: ", function_name);
    
    auto library_name = function_name.substr(0, separator_pos);
    auto kernel_name = function_name.substr(separator_pos + 2u);
    
    LUISA_INFO("Loading kernel: \"", kernel_name, "\", library: \"", library_name, "\"");
    
    auto iter = _function_wrappers.find(function_name);
    if (iter == _function_wrappers.end()) {
        
        auto descriptor = [[MTLComputePipelineDescriptor alloc] init];
        auto function = [_load_library(library_name) newFunctionWithName:make_objc_string(kernel_name)];
        LUISA_ERROR_IF(function == nullptr, "Failed to load kernel: \"", kernel_name, "\", library: \"", library_name, "\"");
        
        descriptor.computeFunction = function;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
        
        MTLAutoreleasedComputePipelineReflection reflection;
        auto pipeline = [_device_wrapper->device newComputePipelineStateWithDescriptor:descriptor
                                                                               options:MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo
                                                                            reflection:&reflection
                                                                                 error:nullptr];
        
        iter = _function_wrappers.emplace(std::string{function_name}, std::make_unique<MetalFunctionWrapper>(function, pipeline, reflection)).first;
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

std::unique_ptr<TypelessBuffer> MetalDevice::allocate_typeless_buffer(size_t capacity, BufferStorage storage) {
    
    auto buffer = [_device_wrapper->device newBufferWithLength:capacity
                                                       options:storage == BufferStorage::DEVICE_PRIVATE ? MTLResourceStorageModePrivate : MTLResourceStorageModeManaged];
    return std::make_unique<MetalBuffer>(buffer, capacity, storage);
}

std::unique_ptr<Acceleration> MetalDevice::build_acceleration(Geometry &geometry) {
    
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

id<MTLLibrary> MetalDevice::_load_library(std::string_view library_name) {
    
    auto library_iter = _library_wrappers.find(library_name);
    if (library_iter == _library_wrappers.end()) {
        
        auto compatibility_header_path = ResourceManager::instance().working_path(serialize("backends/metal/metal_compatibility.h"));
        auto source = text_file_contents(ResourceManager::instance().working_path(serialize("src/kernels/", library_name, ".sk")));
        
        auto cache_dir = ResourceManager::instance().working_path("cache");
        if (!std::filesystem::exists(cache_dir)) { LUISA_WARNING_IF_NOT(std::filesystem::create_directory(cache_dir), "Failed to create cache folder: ", cache_dir); }
        
        auto source_path = cache_dir / serialize(library_name, ".metal");
        LUISA_INFO("Generating Metal shader: ", library_name);
        if (std::ofstream source_file{source_path}; source_file.is_open()) {
            source_file << "#define LUISA_DEVICE_COMPATIBLE\n"
                        << "#include <" << compatibility_header_path.string() << ">\n"
                        << source << std::endl;
        } else { LUISA_EXCEPTION("Failed to create file for Metal shader: ", source_path); }
        
        auto ir_path = cache_dir / serialize(library_name, ".air");
        auto library_path = cache_dir / serialize(library_name, ".metallib");
        auto include_path = ResourceManager::instance().working_path(serialize("src"));
        auto compile_command = serialize("xcrun -sdk macosx metal -Wall -Wextra -Wno-c++17-extensions -O3 -ffast-math -I ", include_path, " -c ", source_path, " -o ", ir_path);
        LUISA_INFO("Compiling Metal source: ", source_path);
        LUISA_EXCEPTION_IF(system(compile_command.c_str()) != 0, "Failed to compile Metal shader source, command: ", compile_command);
        
        auto archive_command = serialize("xcrun -sdk macosx metallib ", ir_path, " -o ", library_path);
        LUISA_INFO("Archiving Metal library: ", ir_path);
        LUISA_EXCEPTION_IF(system(archive_command.c_str()) != 0, "Failed to archive Metal library, command: ", archive_command);
        
        auto library = [_device_wrapper->device newLibraryWithFile:make_objc_string(library_path.string()) error:nullptr];
        LUISA_EXCEPTION_IF(library == nullptr, "Failed to load library: ", library_path);
        library_iter = _library_wrappers.emplace(library_name, std::make_unique<MetalLibraryWrapper>(library)).first;
    }
    return library_iter->second->library;
}

}

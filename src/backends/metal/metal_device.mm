//
// Created by Mike Smith on 2019/10/24.
//

#import <cstdlib>
#import <string>
#import <fstream>
#import <queue>
#import <cstring>
#import <map>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <compute/device.h>
#import <render/geometry.h>

#import "metal_buffer.h"
#import "metal_kernel.h"
#import "metal_acceleration.h"
#import "metal_codegen.h"
#import "metal_dispatcher.h"

namespace luisa::metal {

using compute::Device;
using render::Geometry;

class MetalDevice : public Device {

public:
    struct MTLFunctionWrapper {
        id<MTLComputePipelineState> pipeline_state;
        id<MTLArgumentEncoder> argument_encoder;
        std::map<std::string, uint32_t> argument_to_id;
    };
    
    static constexpr auto max_command_queue_size = 8u;

private:
    id<MTLDevice> _handle;
    id<MTLCommandQueue> _command_queue;
    std::map<DigestSHA1, MTLFunctionWrapper> _kernel_cache;
    std::vector<MetalDispatcher> _dispatchers;
    uint32_t _next_dispatcher{0u};
    
    [[nodiscard]] MetalDispatcher &next_dispatcher() noexcept {
        auto id = _next_dispatcher;
        _next_dispatcher = (_next_dispatcher + 1u) % max_command_queue_size;
        auto &&dispatcher = _dispatchers[id];
        dispatcher.synchronize();
        return dispatcher;
    }

protected:
    std::unique_ptr<Kernel> _compile_kernel(const compute::dsl::Function &f) override;
    std::unique_ptr<Buffer> _allocate_buffer(size_t size) override;

public:
    explicit MetalDevice(Context *context);
    ~MetalDevice() noexcept override = default;
    std::unique_ptr<Acceleration> build_acceleration(Geometry &geometry) override;
    void launch(const std::function<void(Dispatcher &)> &dispatch) override;
    void synchronize() override;
};

MetalDevice::MetalDevice(Context *context) : Device{context} {
    _handle = MTLCreateSystemDefaultDevice();
    _command_queue = [_handle newCommandQueue];
    _dispatchers.resize(max_command_queue_size);
}

std::unique_ptr<Acceleration> MetalDevice::build_acceleration(Geometry &geometry) {
    
    auto acceleration_group = [[MPSAccelerationStructureGroup alloc] initWithDevice:_handle];
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
    instance_acceleration.usage = geometry.dynamic_shapes().empty() && geometry.dynamic_instances().empty() ?
                                  MPSAccelerationStructureUsageNone :
                                  MPSAccelerationStructureUsageRefit;
    
    [instance_acceleration rebuild];
    
    auto ray_intersector = [[MPSRayIntersector alloc] initWithDevice:_handle];
    ray_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    ray_intersector.rayStride = sizeof(Ray);
    ray_intersector.intersectionDataType = MPSIntersectionDataTypeDistancePrimitiveIndexInstanceIndexCoordinates;
    ray_intersector.intersectionStride = sizeof(ClosestHit);
    ray_intersector.boundingBoxIntersectionTestType = MPSBoundingBoxIntersectionTestTypeAxisAligned;
    ray_intersector.triangleIntersectionTestType = MPSTriangleIntersectionTestTypeWatertight;
    
    auto shadow_ray_intersector = [[MPSRayIntersector alloc] initWithDevice:_handle];
    shadow_ray_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    shadow_ray_intersector.rayStride = sizeof(Ray);
    shadow_ray_intersector.intersectionDataType = MPSIntersectionDataTypeDistance;
    shadow_ray_intersector.intersectionStride = sizeof(AnyHit);
    shadow_ray_intersector.boundingBoxIntersectionTestType = MPSBoundingBoxIntersectionTestTypeAxisAligned;
    shadow_ray_intersector.triangleIntersectionTestType = MPSTriangleIntersectionTestTypeWatertight;
    
    return std::make_unique<MetalAcceleration>(acceleration_group, instance_acceleration, ray_intersector, shadow_ray_intersector);
}

std::unique_ptr<Kernel> MetalDevice::_compile_kernel(const compute::dsl::Function &f) {
    
    LUISA_INFO("Generating source for kernel \"", f.name(), "\"...");
    
    std::ostringstream os;
    MetalCodegen codegen{os};
    codegen.emit(f);
    auto s = os.str();
    
    auto digest = sha1_digest(s);
    auto iter = _kernel_cache.find(digest);
    if (iter == _kernel_cache.cend()) {
        
        LUISA_INFO("No compilation cache found for kernel \"", f.name(), "\", compiling from source...");
        NSError *error = nullptr;
        auto library = [_handle newLibraryWithSource:@(s.c_str()) options:nullptr error:&error];
        if (error != nullptr) {
            LUISA_WARNING("Compilation output:");
            NSLog(@"%@", error);
        }
        LUISA_INFO("Compilation for kernel \"", f.name(), "\" succeeded.");
        
        // Create PSO
        auto function = [library newFunctionWithName:@(f.name().c_str())];
        auto desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = function;
        desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
        desc.label = @(f.name().c_str());
        
        error = nullptr;
        auto pso = [_handle newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nullptr error:&error];
        if (error != nullptr) {
            LUISA_WARNING("Error occurred while creating pipeline state object, reason:");
            NSLog(@"%@", error);
            LUISA_EXCEPTION("Failed to create pipeline state object for kernel \"", f.name(), "\".");
        }
        
        MTLAutoreleasedArgument reflection;
        auto argument_encoder = [function newArgumentEncoderWithBufferIndex:0 reflection:&reflection];
        MTLFunctionWrapper wrapper{.pipeline_state = pso, .argument_encoder = argument_encoder};
        
        auto arg_struct = reflection.bufferStructType;
        for (auto &&arg : f.arguments()) {
            auto arg_name = serialize("v", arg.uid());
            auto arg_reflection = [arg_struct memberByName:@(arg_name.c_str())];
            if (compute::dsl::is_ptr_or_ref(arg.type())) {
                LUISA_EXCEPTION_IF([arg_reflection pointerType] == nullptr, "Argument \"", arg_name, "\" bound to a buffer is not declared as a pointer or reference.");
            } else if (arg.type()->type == compute::dsl::TypeCatalog::TEXTURE) {
                LUISA_EXCEPTION_IF([arg_reflection textureReferenceType] == nullptr, "Argument \"", arg_name, "\" bound to a texture is not declared as a texture.");
            }
            wrapper.argument_to_id.emplace(arg_name, arg_reflection.argumentIndex);
        }
        iter = _kernel_cache.emplace(digest, wrapper).first;
    } else {
        LUISA_INFO("Cache hit for kernel \"", f.name(), "\", compilation skipped.");
    }
    
    auto pso = iter->second.pipeline_state;
    auto arg_enc = iter->second.argument_encoder;
    auto &&arg_ids = iter->second.argument_to_id;
    auto arg_buffer = [_handle newBufferWithLength:arg_enc.encodedLength options:MTLResourceOptionCPUCacheModeWriteCombined];
    [arg_enc setArgumentBuffer:arg_buffer offset:0u];
    std::vector<MetalKernel::UniformBinding> uniforms;
    std::vector<MetalKernel::ResourceUsage> resources;
    for (auto &&arg : f.arguments()) {
        auto arg_name = serialize("v", arg.uid());
        auto arg_index = arg_ids.at(arg_name);
        if (arg.is_buffer_argument()) {
            auto usage = compute::dsl::is_const_ptr_or_ref(arg.type()) ? MTLResourceUsageRead : (MTLResourceUsageRead | MTLResourceUsageWrite);
            auto buffer = dynamic_cast<MetalBuffer *>(arg.buffer().buffer())->handle();
            resources.emplace_back(MetalKernel::ResourceUsage{buffer, usage});
            [arg_enc setBuffer:buffer offset:0u atIndex:arg_index];
        } else if (arg.is_texture_argument()) {
            // TODO...
            LUISA_ERROR("Not implemented...");
        } else if (arg.is_immutable_argument()) {
            std::memmove([arg_enc constantDataAtIndex:arg_index], arg.immutable_data().data(), arg.immutable_data().size());
        } else if (arg.is_uniform_argument()) {
            uniforms.emplace_back(MetalKernel::UniformBinding{[arg_enc constantDataAtIndex:arg_index], arg.uniform_data(), arg.type()->size});
        }
    }
    
    return std::make_unique<MetalKernel>(pso, arg_buffer, std::move(uniforms), std::move(resources));
}

void MetalDevice::launch(const std::function<void(Dispatcher &)> &dispatch) {
    auto &&dispatcher = next_dispatcher();
    auto command_buffer = [_command_queue commandBuffer];
    dispatcher.reset(command_buffer);
    dispatch(dispatcher);
    dispatcher.commit();
}

void MetalDevice::synchronize() {
    for (auto i = 0u; i < max_command_queue_size; i++) { next_dispatcher().reset(); }
}

std::unique_ptr<Buffer> MetalDevice::_allocate_buffer(size_t size) {
    auto buffer = [_handle newBufferWithLength:size options:MTLResourceStorageModePrivate];
    return std::make_unique<MetalBuffer>(buffer, size);
}

}

LUISA_EXPORT_DEVICE_CREATOR(luisa::metal::MetalDevice)

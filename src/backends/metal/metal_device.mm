//
// Created by Mike Smith on 2019/10/24.
//

#import <cstdlib>
#import <queue>
#import <cstring>
#import <map>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <core/hash.h>

#import <compute/device.h>
#import <compute/acceleration.h>

#import "metal_buffer.h"
#import "metal_texture.h"
#import "metal_kernel.h"
#import "metal_codegen.h"
#import "metal_dispatcher.h"
#import "metal_acceleration.h"

namespace luisa::metal {

using namespace compute;

class MetalDevice : public Device {

public:
    static constexpr auto max_command_queue_size = 8u;

private:
    id<MTLDevice> _handle;
    id<MTLCommandQueue> _command_queue;
    std::map<SHA1::Digest, id<MTLComputePipelineState>> _kernel_cache;
    std::vector<std::unique_ptr<MetalDispatcher>> _dispatchers;
    uint32_t _next_dispatcher{0u};
    
    [[nodiscard]] MetalDispatcher &_get_next_dispatcher() noexcept {
        auto id = _next_dispatcher;
        _next_dispatcher = (_next_dispatcher + 1u) % max_command_queue_size;
        auto &&dispatcher = *_dispatchers[id];
        dispatcher.wait();
        return dispatcher;
    }

protected:
    std::shared_ptr<Kernel> _compile_kernel(const compute::dsl::Function &f) override;
    std::shared_ptr<Buffer> _allocate_buffer(size_t size) override;
    std::shared_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format) override;
    void _launch(const std::function<void(Dispatcher &)> &dispatch) override;

public:
    explicit MetalDevice(Context *context, uint32_t device_id);
    ~MetalDevice() noexcept override = default;
    void synchronize() override;
    
    std::unique_ptr<Acceleration> build_acceleration(
        const BufferView<float3> &positions,
        const BufferView<packed_uint3> &indices,
        const std::vector<packed_uint3> &meshes,
        const BufferView<uint> &instances,
        const BufferView<float4x4> &transforms,
        bool is_static) override;
};

MetalDevice::MetalDevice(Context *context, uint32_t device_id) : Device{context} {
    auto devices = MTLCopyAllDevices();
    LUISA_ERROR_IF_NOT(device_id < devices.count, "Invalid Metal device index ", device_id, ": max available index is ", devices.count - 1, ".");
    _handle = devices[device_id];
    LUISA_INFO("Created Metal device #", device_id, ", description:\n", [_handle.description cStringUsingEncoding:NSUTF8StringEncoding]);
    _command_queue = [_handle newCommandQueue];
    _dispatchers.reserve(max_command_queue_size);
    for (auto i = 0u; i < max_command_queue_size; i++) {
        _dispatchers.emplace_back(std::make_unique<MetalDispatcher>());
    }
}

std::shared_ptr<Kernel> MetalDevice::_compile_kernel(const compute::dsl::Function &f) {
    
    LUISA_INFO("Generating source for kernel \"", f.name(), "\"...");
    
    std::ostringstream os;
    MetalCodegen codegen{os};
    codegen.emit(f);
    auto s = os.str();
    LUISA_INFO("Generated source:\n", s);
    
    auto digest = SHA1{s}.digest();
    auto iter = _kernel_cache.find(digest);
    if (iter == _kernel_cache.cend()) {
        
        LUISA_INFO("No compilation cache found for kernel \"", f.name(), "\", compiling from source...");
        NSError *error = nullptr;
        auto library = [_handle newLibraryWithSource:@(s.c_str()) options:nullptr error:&error];
        if (error != nullptr) {
            LUISA_WARNING("Compilation output:");
            NSLog(@"%@", error);
        }
        
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
        LUISA_INFO("Compilation for kernel \"", f.name(), "\" succeeded.");
        iter = _kernel_cache.emplace(digest, pso).first;
    } else {
        LUISA_INFO("Cache hit for kernel \"", f.name(), "\", compilation skipped.");
    }
    
    auto pso = iter->second;
    
    std::vector<MetalKernel::Uniform> uniforms;
    std::vector<MetalKernel::Resource> resources;
    
    size_t uniform_offset = 0u;
    for (auto &&arg : f.arguments()) {
        if (arg->is_buffer_argument()) {
            MetalKernel::Resource r;
            r.buffer = arg->buffer()->shared_from_this();
            resources.emplace_back(std::move(r));
        } else if (arg->is_texture_argument()) {
            MetalKernel::Resource r;
            r.texture = arg->texture()->shared_from_this();
            resources.emplace_back(std::move(r));
        } else if (arg->is_uniform_argument()) {
            auto alignment = arg->type()->alignment;
            uniform_offset = (uniform_offset + alignment - 1u) / alignment * alignment;
            MetalKernel::Uniform uniform;
            uniform.offset = uniform_offset;
            uniform.binding = arg->uniform_data();
            uniform.binding_size = arg->type()->size;
            uniforms.emplace_back(std::move(uniform));
            uniform_offset += arg->type()->size;
        } else if (arg->is_immutable_argument()) {
            auto alignment = arg->type()->alignment;
            uniform_offset = (uniform_offset + alignment - 1u) / alignment * alignment;
            MetalKernel::Uniform uniform;
            uniform.immutable = arg->immutable_data();
            uniform.offset = uniform_offset;
            uniforms.emplace_back(std::move(uniform));
            uniform_offset += arg->type()->size;
        }
    }
    return std::make_shared<MetalKernel>(pso, std::move(resources), std::move(uniforms));
}

void MetalDevice::_launch(const std::function<void(Dispatcher &)> &dispatch) {
    auto &&dispatcher = _get_next_dispatcher();
    auto command_buffer = [_command_queue commandBuffer];
    dispatcher.reset(command_buffer);
    dispatch(dispatcher);
    dispatcher.commit();
}

void MetalDevice::synchronize() {
    for (auto i = 0u; i < max_command_queue_size; i++) { _get_next_dispatcher().reset(); }
}

std::shared_ptr<Buffer> MetalDevice::_allocate_buffer(size_t size) {
    auto buffer = [_handle newBufferWithLength:size options:MTLResourceStorageModePrivate];
    return std::make_shared<MetalBuffer>(buffer, size);
}

std::shared_ptr<Texture> MetalDevice::_allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format) {
    
    auto desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2D;
    desc.width = width;
    desc.height = height;
    desc.allowGPUOptimizedContents = true;
    desc.storageMode = MTLStorageModePrivate;
    switch (format) {
        case PixelFormat::R8U:
            desc.pixelFormat = MTLPixelFormatA8Unorm;
            break;
        case PixelFormat::RG8U:
            desc.pixelFormat = MTLPixelFormatRG8Unorm;
            break;
        case PixelFormat::RGBA8U:
            desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case PixelFormat::R32F:
            desc.pixelFormat = MTLPixelFormatR32Float;
            break;
        case PixelFormat::RG32F:
            desc.pixelFormat = MTLPixelFormatRG32Float;
            break;
        case PixelFormat::RGBA32F:
            desc.pixelFormat = MTLPixelFormatRGBA32Float;
            break;
    }
    auto texture = [_handle newTextureWithDescriptor:desc];
    return std::make_shared<MetalTexture>(texture, width, height, format);
}

std::unique_ptr<Acceleration> MetalDevice::build_acceleration(
    const BufferView<float3> &positions,
    const BufferView<packed_uint3> &indices,
    const std::vector<packed_uint3> &meshes,
    const BufferView<uint> &instances,
    const BufferView<float4x4> &transforms,
    bool is_static) {
    
    auto acceleration_group = [[MPSAccelerationStructureGroup alloc] initWithDevice:_handle];
    auto instance_acceleration = [[MPSInstanceAccelerationStructure alloc] initWithGroup:acceleration_group];
    
    auto acceleration_structures = [NSMutableArray array];
    instance_acceleration.accelerationStructures = acceleration_structures;
    
    // create individual triangle acceleration structures
    for (auto [vertex_offset, triangle_offset, triangle_count] : meshes) {
        auto triangle_acceleration = [[MPSTriangleAccelerationStructure alloc] initWithGroup:acceleration_group];
        triangle_acceleration.vertexBuffer = dynamic_cast<MetalBuffer *>(positions.buffer())->handle();
        triangle_acceleration.vertexBufferOffset = positions.byte_offset() + vertex_offset * sizeof(float3);
        triangle_acceleration.vertexStride = sizeof(float3);
        triangle_acceleration.indexBuffer = dynamic_cast<MetalBuffer *>(indices.buffer())->handle();
        triangle_acceleration.indexBufferOffset = indices.byte_offset() + triangle_offset * sizeof(packed_uint3);
        triangle_acceleration.indexType = MPSDataTypeUInt32;
        triangle_acceleration.triangleCount = triangle_count;
        [triangle_acceleration rebuild];
        [acceleration_structures addObject:triangle_acceleration];
    }
    
    instance_acceleration.instanceBuffer = dynamic_cast<MetalBuffer *>(instances.buffer())->handle();
    instance_acceleration.instanceBufferOffset = instances.byte_offset();
    instance_acceleration.instanceCount = instances.size();
    instance_acceleration.transformBuffer = dynamic_cast<MetalBuffer *>(transforms.buffer())->handle();
    instance_acceleration.transformBufferOffset = transforms.byte_offset();
    instance_acceleration.transformType = MPSTransformTypeFloat4x4;
    instance_acceleration.usage = is_static ? MPSAccelerationStructureUsageNone : MPSAccelerationStructureUsageRefit;
    [instance_acceleration rebuild];
    
    auto closest_intersector = [[MPSRayIntersector alloc] initWithDevice:_handle];
    closest_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    closest_intersector.rayStride = sizeof(Ray);
    closest_intersector.intersectionDataType = MPSIntersectionDataTypeDistancePrimitiveIndexInstanceIndexCoordinates;
    closest_intersector.intersectionStride = sizeof(ClosestHit);
    closest_intersector.boundingBoxIntersectionTestType = MPSBoundingBoxIntersectionTestTypeAxisAligned;
    closest_intersector.triangleIntersectionTestType = MPSTriangleIntersectionTestTypeWatertight;
    
    auto any_intersector = [[MPSRayIntersector alloc] initWithDevice:_handle];
    any_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    any_intersector.rayStride = sizeof(Ray);
    any_intersector.intersectionDataType = MPSIntersectionDataTypeDistance;
    any_intersector.intersectionStride = sizeof(AnyHit);
    any_intersector.boundingBoxIntersectionTestType = MPSBoundingBoxIntersectionTestTypeAxisAligned;
    any_intersector.triangleIntersectionTestType = MPSTriangleIntersectionTestTypeWatertight;
    
    return std::make_unique<MetalAcceleration>(instance_acceleration, closest_intersector, any_intersector);
}

}

LUISA_EXPORT_DEVICE_CREATOR(luisa::metal::MetalDevice)

//
// Created by Mike on 9/20/2020.
//

#ifdef LUISA_OPTIX_AVAILABLE

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "cuda_acceleration.h"
#include "cuda_device.h"

#define OPTIX_CHECK(call)                                                      \
    [&] {                                                                      \
        auto res = call;                                                       \
        LUISA_EXCEPTION_IF_NOT(                                                \
            res == OPTIX_SUCCESS,                                              \
            "OptiX call [ ", #call, " ] failed: ", optixGetErrorString(res));  \
    }()
    
#include <compute/dsl_syntax.h>

namespace luisa::cuda {

using namespace compute;
using namespace compute::dsl;

CudaAcceleration::CudaAcceleration(
    CudaDevice *device,
    const BufferView<luisa::float3> &positions,
    const BufferView<TriangleHandle> &indices,
    const std::vector<MeshHandle> &meshes,
    const BufferView<uint> &instances,
    const BufferView<luisa::float4x4> &transforms,
    bool is_static) {
    
    OPTIX_CHECK(optixInit());
    
    OptixDeviceContextOptions options = {};
    OPTIX_CHECK(optixDeviceContextCreate(device->context(), &options, &_optix_ctx));
    
    auto compacted_size_buffer = device->allocate_buffer<uint2>(1u);
    
    // create geometry acceleration structures
    for (auto mesh : meshes) {
        
        OptixAccelBuildOptions accel_options{};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        
        OptixBuildInput triangle_input{};
        
        constexpr uint32_t triangle_input_flag = OPTIX_GEOMETRY_FLAG_NONE;
        triangle_input.triangleArray.flags = &triangle_input_flag;
        triangle_input.triangleArray.numSbtRecords = 1;
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        
        auto vertex_buffer = dynamic_cast<CudaBuffer *>(positions.buffer())->handle() + positions.byte_offset() + mesh.vertex_offset * sizeof(luisa::float3);
        triangle_input.triangleArray.vertexBuffers = &vertex_buffer;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(luisa::float3);
        triangle_input.triangleArray.numVertices = mesh.vertex_count;
        
        triangle_input.triangleArray.numIndexTriplets = mesh.triangle_count;
        triangle_input.triangleArray.indexBuffer =
            dynamic_cast<CudaBuffer *>(indices.buffer())->handle() + indices.byte_offset() + mesh.triangle_offset * sizeof(TriangleHandle);
        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes = sizeof(TriangleHandle);
        
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(_optix_ctx, &accel_options, &triangle_input, 1, &gas_buffer_sizes));
        
        // Allocate device memory for the scratch space buffer as well
        // as the GAS itself
        auto gas_temp_buffer = device->allocate_buffer<uchar>(gas_buffer_sizes.tempSizeInBytes);
        auto gas_output_buffer = device->allocate_buffer<uchar>(gas_buffer_sizes.outputSizeInBytes);
        
        OptixTraversableHandle gas_handle = 0u;
        
        OptixAccelEmitDesc emit_desc;
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = dynamic_cast<CudaBuffer *>(compacted_size_buffer.buffer())->handle();
        
        size_t compacted_size = 0u;
        device->launch([&](Dispatcher &dispatch) {
            auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
            OPTIX_CHECK(optixAccelBuild(_optix_ctx, stream, &accel_options, &triangle_input, 1,
                                        dynamic_cast<CudaBuffer *>(gas_temp_buffer.buffer())->handle() + gas_temp_buffer.byte_offset(), gas_buffer_sizes.tempSizeInBytes,
                                        dynamic_cast<CudaBuffer *>(gas_output_buffer.buffer())->handle() + gas_output_buffer.byte_offset(), gas_buffer_sizes.outputSizeInBytes,
                                        &gas_handle, &emit_desc, 1));
            dispatch(compacted_size_buffer.copy_to(&compacted_size));
        });
        device->synchronize();
        
        auto gas_buffer = device->allocate_buffer<uchar>(compacted_size);
        device->launch([&](Dispatcher &dispatch) {
            auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
            OPTIX_CHECK(optixAccelCompact(_optix_ctx, stream, gas_handle,
                                          dynamic_cast<CudaBuffer *>(gas_buffer.buffer())->handle() + gas_buffer.byte_offset(),
                                          compacted_size, &gas_handle));
        });
        device->synchronize();
        
        _gas_handles.emplace_back(gas_handle);
        _gas_buffers.emplace_back(std::move(gas_buffer));
    }
    
    _gas_handle_buffer = device->allocate_buffer<Traversable>(_gas_handles.size());
    _instance_buffer = device->allocate_buffer<Instance>(instances.size());
    
    auto instance_count = static_cast<uint>(instances.size());
    auto initialize_instance_buffer_kernel = device->compile_kernel("init_instance_buffer", [&] {
        auto tid = thread_id();
        If (tid < instance_count) {
            Var transform = transpose(transforms[tid]);
            auto instance = _instance_buffer[tid];
            instance.transform[0u] = transform[0u];
            instance.transform[1u] = transform[1u];
            instance.transform[2u] = transform[2u];
            instance.instance_id = tid;
            instance.sbt_offset = 0u;
            instance.mask = 0xffffffffu;
            instance.flags = static_cast<uint>(OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
            instance.traversable = _gas_handle_buffer[tid];
        };
    });
    
    device->launch([&](Dispatcher &dispatch) {
        dispatch(_gas_handle_buffer.copy_from(_gas_handles.data()));
        dispatch(initialize_instance_buffer_kernel.parallelize(instance_count));
    });
    
    
    // create instance acceleration structure
    OptixAccelBuildOptions build_options{};
    build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (!is_static) { build_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE; }
    build_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixBuildInput build_input{};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
}

void CudaAcceleration::_refit(Dispatcher &dispatch) {

}

void CudaAcceleration::_intersect_any(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer) const {

}

void CudaAcceleration::_intersect_closest(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer) const {

}

}

#endif

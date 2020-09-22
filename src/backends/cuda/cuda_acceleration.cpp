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

namespace luisa::cuda {

using namespace compute;
using namespace compute::dsl;

CudaAcceleration::CudaAcceleration(
    CudaDevice *device,
    const BufferView<float3> &positions,
    const BufferView<TriangleHandle> &indices,
    const std::vector<MeshHandle> &meshes,
    const BufferView<uint> &instances,
    const BufferView<float4x4> &transforms,
    bool is_static) {
    
    OPTIX_CHECK(optixInit());
    
    OptixDeviceContextOptions options = {};
    OPTIX_CHECK(optixDeviceContextCreate(device->context(), &options, &_optix_ctx));
    
    auto compacted_size_buffer = device->allocate_buffer<uint2>(1u);
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
        CUdeviceptr d_temp_buffer_gas;
        CUdeviceptr d_gas_output_buffer;
        CUDA_CHECK(cuMemAlloc(&d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cuMemAlloc(&d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes));
        
        OptixTraversableHandle gas_handle = 0u;
        
        OptixAccelEmitDesc emit_desc;
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = dynamic_cast<CudaBuffer *>(compacted_size_buffer.buffer())->handle();
        
        size_t compacted_size = 0u;
        device->launch([&](Dispatcher &dispatch) {
            auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
            OPTIX_CHECK(optixAccelBuild(_optix_ctx, stream, &accel_options, &triangle_input, 1,
                                        d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                        d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes,
                                        &gas_handle, &emit_desc, 1));
            dispatch(compacted_size_buffer.copy_to(&compacted_size));
        });
        device->synchronize();
        
        CUdeviceptr gas_buffer;
        CUDA_CHECK(cuMemAlloc(&gas_buffer, compacted_size));
        device->launch([&](Dispatcher &dispatch) {
            auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
            OPTIX_CHECK(optixAccelCompact(_optix_ctx, stream, gas_handle, gas_buffer, compacted_size, &gas_handle));
            CUDA_CHECK(cuMemFree(d_temp_buffer_gas));
            CUDA_CHECK(cuMemFree(d_gas_output_buffer));
        });
        device->synchronize();
        
        _gas_handles.emplace_back(gas_handle);
        _gas_buffers.emplace_back(gas_buffer);
    }
    
    
}

void CudaAcceleration::_refit(Dispatcher &dispatch) {

}

void CudaAcceleration::_intersect_any(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer) const {

}

void CudaAcceleration::_intersect_closest(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer) const {

}

}

#endif

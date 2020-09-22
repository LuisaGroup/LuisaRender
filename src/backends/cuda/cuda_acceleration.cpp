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
    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(device->context(), &options, &context));
    
    std::vector<OptixTraversableHandle> geometry_accels;
    
    for (auto mesh : meshes) {
        
        OptixAccelBuildOptions accel_options{};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    
        constexpr uint32_t triangle_input_flags[] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput triangle_input{};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(luisa::float3);
        triangle_input.triangleArray.numVertices = mesh.vertex_count;
        
        auto vertex_buffer = dynamic_cast<CudaBuffer *>(positions.buffer())->handle() + positions.byte_offset() + mesh.vertex_offset * sizeof(luisa::float3);
//        triangle_input.triangleArray.vertexBuffers =
//        triangle_input.triangleArray.numIndexTriplets =
//        triangle_input.triangleArray.vertexBuffers = &d_vertices;
//        triangle_input.triangleArray.flags = triangle_input_flags;
//        triangle_input.triangleArray.numSbtRecords = 1;
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

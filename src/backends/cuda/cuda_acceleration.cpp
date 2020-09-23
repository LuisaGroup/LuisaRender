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
    
#define OPTIX_CHECK_LOG(call)                                                  \
    [&] {                                                                      \
        auto res = call;                                                       \
        LUISA_EXCEPTION_IF_NOT(                                                \
            res == OPTIX_SUCCESS,                                              \
            "OptiX call [ ", #call, " ] failed: ", optixGetErrorString(res),   \
            "    \nLog: ", std::string_view{log});                             \
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
    bool is_static)
    : _device{device},
      _instance_transform_buffer{transforms} {
    
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
            Var<Instance> instance;
            instance.transform[0u] = transform[0u];
            instance.transform[1u] = transform[1u];
            instance.transform[2u] = transform[2u];
            instance.instance_id = tid;
            instance.sbt_offset = 0u;
            instance.mask = 0xffu;
            instance.flags = static_cast<uint>(OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
            instance.traversable = _gas_handle_buffer[tid];
            _instance_buffer[tid] = instance;
        };
    });
    
    device->launch([&](Dispatcher &dispatch) {
        dispatch(_gas_handle_buffer.copy_from(_gas_handles.data()));
        dispatch(initialize_instance_buffer_kernel.parallelize(instance_count));
    });
    device->synchronize();
    
    // create instance acceleration structure
    OptixAccelBuildOptions build_options{};
    build_options.buildFlags = is_static ?
                               OPTIX_BUILD_FLAG_PREFER_FAST_TRACE :
                               (OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    build_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    build_options.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
    build_options.motionOptions.numKeys = 0u;
    
    OptixBuildInput build_input{};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = dynamic_cast<CudaBuffer *>(_instance_buffer.buffer())->handle() + _instance_buffer.byte_offset();
    build_input.instanceArray.numInstances = instance_count;
    
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(_optix_ctx, &build_options, &build_input, 1, &buffer_sizes));
    LUISA_INFO("BufferSizes: tempSizeInBytes = ", buffer_sizes.tempSizeInBytes, ", ",
               "outputSizeInBytes = ", buffer_sizes.outputSizeInBytes, ", ",
               "tempUpdateSizeInBytes = ", buffer_sizes.tempUpdateSizeInBytes);
    
    auto temp_buffer = device->allocate_buffer<uchar>(buffer_sizes.tempSizeInBytes);
    _ias_buffer = device->allocate_buffer<uchar>(buffer_sizes.outputSizeInBytes);
    
    device->launch([&](Dispatcher &dispatch) {
        auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
        OPTIX_CHECK(optixAccelBuild(_optix_ctx, stream, &build_options, &build_input, 1,
                                    dynamic_cast<CudaBuffer *>(temp_buffer.buffer())->handle() + temp_buffer.byte_offset(), buffer_sizes.tempSizeInBytes,
                                    dynamic_cast<CudaBuffer *>(_ias_buffer.buffer())->handle() + _ias_buffer.byte_offset(), buffer_sizes.outputSizeInBytes,
                                    &_ias_handle, nullptr, 0));
    });
    device->synchronize();
    _gas_handle_buffer.clear_cache();
    LUISA_INFO("Successfully build OptiX acceleration structure.");
    
    // Create OptiX module and pipeline
//    OptixModule module = nullptr;
//    OptixPipelineCompileOptions pipeline_compile_options = {};
//    OptixModuleCompileOptions module_compile_options = {};
//    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
//    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
//    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
//
//    pipeline_compile_options.usesMotionBlur = false;
//    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
//    pipeline_compile_options.numPayloadValues = 2;
//    pipeline_compile_options.numAttributeValues = 2;
//    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
//    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
//
//    auto ptx = text_file_contents(static_cast<Device *>(device)->context().runtime_path("bin") / "backends" / "optix_adapter.ptx");
//
//    size_t log_size = 4096u;
//    char log[log_size];
//
//    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(_optix_ctx, &module_compile_options, &pipeline_compile_options,
//                                         ptx.c_str(), ptx.size(), log, &log_size, &module));
//
//    // Create program groups, including NULL miss and hitgroups
//    OptixProgramGroup raygen_prog_group = nullptr;
//    OptixProgramGroup miss_prog_group = nullptr;
//    OptixProgramGroup hitgroup_prog_group = nullptr;
//    {
//        OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
//        OptixProgramGroupDesc raygen_prog_group_desc = {};
//        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
//        raygen_prog_group_desc.raygen.module = module;
//        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__trace";
//        size_t sizeof_log = sizeof(log);
//        OPTIX_CHECK_LOG(optixProgramGroupCreate(
//            context,
//            &raygen_prog_group_desc,
//            1,   // num program groups
//            &program_group_options,
//            log,
//            &sizeof_log,
//            &raygen_prog_group
//        ));
//
//        // Leave miss group's module and entryfunc name null
//        OptixProgramGroupDesc miss_prog_group_desc = {};
//        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
//        sizeof_log = sizeof(log);
//        OPTIX_CHECK_LOG(optixProgramGroupCreate(
//            context,
//            &miss_prog_group_desc,
//            1,   // num program groups
//            &program_group_options,
//            log,
//            &sizeof_log,
//            &miss_prog_group
//        ));
//
//        // Leave hit group's module and entryfunc name null
//        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
//        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
//        sizeof_log = sizeof(log);
//        OPTIX_CHECK_LOG(optixProgramGroupCreate(
//            context,
//            &hitgroup_prog_group_desc,
//            1,   // num program groups
//            &program_group_options,
//            log,
//            &sizeof_log,
//            &hitgroup_prog_group
//        ));
//    }
//
//    //
//    // Link pipeline
//    //
//    OptixPipeline pipeline = nullptr;
//    {
//        const uint32_t max_trace_depth = 0;
//        OptixProgramGroup program_groups[] = {raygen_prog_group};
//
//        OptixPipelineLinkOptions pipeline_link_options = {};
//        pipeline_link_options.maxTraceDepth = max_trace_depth;
//        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
//        size_t sizeof_log = sizeof(log);
//        OPTIX_CHECK_LOG(optixPipelineCreate(
//            context,
//            &pipeline_compile_options,
//            &pipeline_link_options,
//            program_groups,
//            sizeof(program_groups) / sizeof(program_groups[0]),
//            log,
//            &sizeof_log,
//            &pipeline
//        ));
//
//        OptixStackSizes stack_sizes = {};
//        for (auto &prog_group : program_groups) {
//            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
//        }
//
//        uint32_t direct_callable_stack_size_from_traversal;
//        uint32_t direct_callable_stack_size_from_state;
//        uint32_t continuation_stack_size;
//        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
//                                               0,  // maxCCDepth
//                                               0,  // maxDCDEpth
//                                               &direct_callable_stack_size_from_traversal,
//                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
//        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
//                                              direct_callable_stack_size_from_state, continuation_stack_size,
//                                              2  // maxTraversableDepth
//        ));
//    }
//
//    //
//    // Set up shader binding table
//    //
//    OptixShaderBindingTable sbt = {};
//    {
//        CUdeviceptr raygen_record;
//        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &raygen_record ), raygen_record_size));
//        RayGenSbtRecord rg_sbt;
//        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
//        rg_sbt.data = {0.462f, 0.725f, 0.f};
//        CUDA_CHECK(cudaMemcpy(
//            reinterpret_cast<void *>( raygen_record ),
//            &rg_sbt,
//            raygen_record_size,
//            cudaMemcpyHostToDevice
//        ));
//
//        CUdeviceptr miss_record;
//        size_t miss_record_size = sizeof(MissSbtRecord);
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &miss_record ), miss_record_size));
//        RayGenSbtRecord ms_sbt;
//        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
//        CUDA_CHECK(cudaMemcpy(
//            reinterpret_cast<void *>( miss_record ),
//            &ms_sbt,
//            miss_record_size,
//            cudaMemcpyHostToDevice
//        ));
//
//        CUdeviceptr hitgroup_record;
//        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &hitgroup_record ), hitgroup_record_size));
//        RayGenSbtRecord hg_sbt;
//        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
//        CUDA_CHECK(cudaMemcpy(
//            reinterpret_cast<void *>( hitgroup_record ),
//            &hg_sbt,
//            hitgroup_record_size,
//            cudaMemcpyHostToDevice
//        ));
//
//        sbt.raygenRecord = raygen_record;
//        sbt.missRecordBase = miss_record;
//        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
//        sbt.missRecordCount = 1;
//        sbt.hitgroupRecordBase = hitgroup_record;
//        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
//        sbt.hitgroupRecordCount = 1;
//    }
//
}

void CudaAcceleration::_refit(Dispatcher &dispatch) {
    
    auto instance_count = static_cast<uint>(_instance_buffer.size());
    if (_instance_update_kernel.empty()) {
        _instance_update_kernel = _device->compile_kernel("cuda_accel_update_instance_buffer", [&] {
            auto tid = thread_id();
            If (tid < instance_count) {
                Var transform = transpose(_instance_transform_buffer[tid]);
                auto instance = _instance_buffer[tid];
                instance.transform[0u] = transform[0u];
                instance.transform[1u] = transform[1u];
                instance.transform[2u] = transform[2u];
            };
        });
    }
    _device->launch(_instance_update_kernel.parallelize(instance_count));
    _device->synchronize();
    
    OptixAccelBuildOptions build_options{};
    build_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    build_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    
    OptixBuildInput build_input{};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = dynamic_cast<CudaBuffer *>(_instance_buffer.buffer())->handle() + _instance_buffer.byte_offset();
    build_input.instanceArray.numInstances = instance_count;
    
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(_optix_ctx, &build_options, &build_input, 1, &buffer_sizes));
    LUISA_INFO("BufferSizes: tempSizeInBytes = ", buffer_sizes.tempSizeInBytes, ", ",
               "outputSizeInBytes = ", buffer_sizes.outputSizeInBytes, ", ",
               "tempUpdateSizeInBytes = ", buffer_sizes.tempUpdateSizeInBytes);
    
    if (_instance_update_buffer.size() < buffer_sizes.tempUpdateSizeInBytes) {
        _instance_update_buffer = _device->allocate_buffer<uchar>(buffer_sizes.tempUpdateSizeInBytes);
    }
    
    auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
    OPTIX_CHECK(optixAccelBuild(_optix_ctx, stream, &build_options, &build_input, 1,
                                dynamic_cast<CudaBuffer *>(_instance_update_buffer.buffer())->handle() + _instance_update_buffer.byte_offset(), buffer_sizes.tempSizeInBytes,
                                dynamic_cast<CudaBuffer *>(_ias_buffer.buffer())->handle() + _ias_buffer.byte_offset(), buffer_sizes.outputSizeInBytes,
                                &_ias_handle, nullptr, 0));
}

void CudaAcceleration::_intersect_any(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer) const {

}

void CudaAcceleration::_intersect_closest(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer) const {

}

}

#endif

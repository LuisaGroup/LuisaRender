//
// Created by Mike on 9/20/2020.
//

#ifdef LUISA_OPTIX_AVAILABLE

#include "cuda_acceleration.h"
#include "cuda_device.h"

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
    bool is_static) : _device{device}, _is_static{is_static}, _input_transform_buffer{transforms} {
    
    _optix_context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
    _optix_context->setCudaDeviceNumbers({device->index()});
    
    for (auto mesh : meshes) {
        auto model = _optix_context->createModel();
        model->setTriangles(
            mesh.triangle_count, RTP_BUFFER_TYPE_CUDA_LINEAR,
            reinterpret_cast<const void *>(dynamic_cast<CudaBuffer *>(indices.buffer())->handle() + indices.byte_offset() + mesh.triangle_offset * sizeof(TriangleHandle)),
            mesh.vertex_count, RTP_BUFFER_TYPE_CUDA_LINEAR,
            reinterpret_cast<const void *>(dynamic_cast<CudaBuffer *>(positions.buffer())->handle() + positions.byte_offset() + mesh.vertex_offset * sizeof(luisa::float3)),
            sizeof(luisa::float3));
        model->setBuilderParameter(RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES, 1);
        model->update(RTP_MODEL_HINT_ASYNC);
        _optix_geometry_models.emplace_back(model);
    }
    
    auto instance_count = static_cast<uint>(instances.size());
    std::vector<uint> gas_instances(instance_count);
    device->launch(instances.copy_to(gas_instances.data()));
    device->synchronize();
    instances.clear_cache();
    
    _optix_geometry_instances.reserve(instance_count);
    for (auto i : gas_instances) {
        _optix_geometry_instances.emplace_back(_optix_geometry_models[gas_instances[i]]->getRTPmodel());
    }
    
    _optix_instance_model = _optix_context->createModel();
    std::vector<RTPmodel> gas_handles;
    gas_handles.reserve(_optix_geometry_models.size());
    for (auto &&model : _optix_geometry_models) { gas_handles.emplace_back(model->getRTPmodel()); }
    
    _optix_transform_buffer = _device->allocate_buffer<std::array<float4, 3>>(instance_count);
    _update_transforms_kernel = _device->compile_kernel("cuda_accel_update_transforms", [&] {
        auto tid = thread_id();
        If (tid < instance_count) {
            Var transform = transpose(_input_transform_buffer[tid]);
            _optix_transform_buffer[tid][0] = transform[0];
            _optix_transform_buffer[tid][1] = transform[1];
            _optix_transform_buffer[tid][2] = transform[2];
        };
    });
    _device->launch(_update_transforms_kernel.parallelize(instance_count));
    
    _optix_instance_model->setInstances(
        instance_count, RTP_BUFFER_TYPE_HOST,
        _optix_geometry_instances.data(),
        RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_CUDA_LINEAR,
        reinterpret_cast<const void *>(dynamic_cast<CudaBuffer *>(_optix_transform_buffer.buffer())->handle() + _optix_transform_buffer.byte_offset()));
    _optix_instance_model->update(RTP_MODEL_HINT_ASYNC);
    
    _optix_anyhit_query = _optix_instance_model->createQuery(RTP_QUERY_TYPE_ANY);
    _optix_closesthit_query = _optix_instance_model->createQuery(RTP_QUERY_TYPE_CLOSEST);
}

void CudaAcceleration::_refit(Dispatcher &dispatch) {
    if (_is_static) { LUISA_WARNING("Ignored refit request on static acceleration structure."); }
    else {
        auto instance_count = static_cast<uint>(_optix_transform_buffer.size());
        dispatch(_update_transforms_kernel.parallelize(instance_count));
        _optix_instance_model->setInstances(
            instance_count, RTP_BUFFER_TYPE_HOST,
            _optix_geometry_instances.data(),
            RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_CUDA_LINEAR,
            reinterpret_cast<const void *>(dynamic_cast<CudaBuffer *>(_optix_transform_buffer.buffer())->handle() + _optix_transform_buffer.byte_offset()));
        _optix_instance_model->update(RTP_MODEL_HINT_ASYNC);
    }
}

void CudaAcceleration::_intersect_any(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer) const {
    
    auto rays_ptr = reinterpret_cast<void *>(dynamic_cast<CudaBuffer *>(ray_buffer.buffer())->handle() + ray_buffer.byte_offset());
    auto hits_ptr = reinterpret_cast<void *>(dynamic_cast<CudaBuffer *>(hit_buffer.buffer())->handle() + hit_buffer.byte_offset());
    _optix_anyhit_query->setRays(ray_buffer.size(), RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, rays_ptr);
    _optix_anyhit_query->setHits(ray_buffer.size(), RTP_BUFFER_FORMAT_HIT_T, RTP_BUFFER_TYPE_CUDA_LINEAR, hits_ptr);
    
    auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
    _optix_anyhit_query->setCudaStream(stream);
    _optix_anyhit_query->execute(RTP_QUERY_HINT_ASYNC);
}

void CudaAcceleration::_intersect_closest(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer) const {
    
    if (_optix_closesthit_buffer.size() < ray_buffer.size()) {
        _optix_closesthit_buffer = _device->allocate_buffer<CudaClosestHit>(ray_buffer.size());
    }
    
    auto rays_ptr = reinterpret_cast<void *>(dynamic_cast<CudaBuffer *>(ray_buffer.buffer())->handle() + ray_buffer.byte_offset());
    auto hits_ptr = reinterpret_cast<void *>(dynamic_cast<CudaBuffer *>(_optix_closesthit_buffer.buffer())->handle() + _optix_closesthit_buffer.byte_offset());
    _optix_closesthit_query->setRays(ray_buffer.size(), RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, rays_ptr);
    _optix_closesthit_query->setHits(ray_buffer.size(), RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR, hits_ptr);
    
    auto stream = dynamic_cast<CudaDispatcher &>(dispatch).handle();
    _optix_closesthit_query->setCudaStream(stream);
    _optix_closesthit_query->execute(RTP_QUERY_HINT_ASYNC);
    
    constexpr auto tg_size = 1024u;
    auto ray_count = static_cast<uint>(ray_buffer.size());
    auto kernel = _device->compile_kernel("cuda_accel_adapt_closest_hits", [&] {
        auto tid = thread_id();
        If (ray_count % tg_size == 0u || tid < ray_count) {
            Var hit = _optix_closesthit_buffer[tid];
            hit_buffer[tid].distance = hit.distance;
            hit_buffer[tid].triangle_id = hit.triangle_id;
            hit_buffer[tid].instance_id = hit.instance_id;
            hit_buffer[tid].bary = make_float2(hit.u, hit.v);
        };
    });
    dispatch(kernel.parallelize(ray_count, tg_size));
}

}

#endif

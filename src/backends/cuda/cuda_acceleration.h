//
// Created by Mike on 9/20/2020.
//

#pragma once

#ifdef LUISA_OPTIX_AVAILABLE

#include <compute/ray.h>
#include <compute/hit.h>
#include <compute/primitive.h>
#include <compute/acceleration.h>
#include <compute/kernel.h>

#include <optix_prime/optix_primepp.h>

namespace luisa::cuda {

struct CudaClosestHit {
    float distance;
    uint triangle_id;
    uint instance_id;
    float u;
    float v;
};

}

LUISA_STRUCT(luisa::cuda::CudaClosestHit, distance, triangle_id, instance_id, u, v)

namespace luisa::cuda {

using compute::Acceleration;
using compute::Buffer;
using compute::BufferView;
using compute::KernelView;
using compute::TriangleHandle;
using compute::MeshHandle;
using compute::Ray;
using compute::AnyHit;
using compute::ClosestHit;

class CudaDevice;

class CudaAcceleration : public Acceleration {

private:
    CudaDevice *_device;
    bool _is_static;
    optix::prime::Context _optix_context;
    std::vector<optix::prime::Model> _optix_geometry_models;
    std::vector<RTPmodel> _optix_geometry_instances;
    optix::prime::Model _optix_instance_model;
    optix::prime::Query _optix_anyhit_query;
    optix::prime::Query _optix_closesthit_query;
    mutable BufferView<CudaClosestHit> _optix_closesthit_buffer;
    BufferView<luisa::float4x4> _input_transform_buffer;
    BufferView<std::array<float4, 3>> _optix_transform_buffer;
    KernelView _update_transforms_kernel;
    mutable KernelView _adapt_interactions_kernel;
    mutable size_t _prev_ray_buffer_offset{0u};
    mutable size_t _prev_hit_buffer_offset{0u};
    mutable Buffer *_prev_ray_buffer{nullptr};
    mutable Buffer *_prev_hit_buffer{nullptr};
    mutable Buffer *_prev_internal_hit_buffer{nullptr};

private:
    void _refit(compute::Dispatcher &dispatch) override;
    void _intersect_any(compute::Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer) const override;
    void _intersect_closest(compute::Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer) const override;

public:
    CudaAcceleration(CudaDevice *device,
                     const BufferView<luisa::float3> &positions,
                     const BufferView<TriangleHandle> &indices,
                     const std::vector<MeshHandle> &meshes,
                     const BufferView<luisa::uint> &instances,
                     const BufferView<luisa::float4x4> &transforms,
                     bool is_static);
    
    ~CudaAcceleration() noexcept override = default;
};

}

#endif

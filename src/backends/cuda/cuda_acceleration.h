//
// Created by Mike on 9/20/2020.
//

#pragma once

#ifdef LUISA_OPTIX_AVAILABLE

#include <optix.h>

#include <compute/ray.h>
#include <compute/hit.h>
#include <compute/primitive.h>
#include <compute/acceleration.h>

namespace luisa::cuda {

using compute::Acceleration;
using compute::BufferView;
using compute::TriangleHandle;
using compute::MeshHandle;
using compute::Ray;
using compute::AnyHit;
using compute::ClosestHit;

class CudaDevice;

class CudaAcceleration : public Acceleration {

private:
    void _refit(compute::Dispatcher &dispatch) override;
    void _intersect_any(compute::Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer) const override;
    void _intersect_closest(compute::Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer) const override;

public:
    CudaAcceleration(CudaDevice *device,
                     const BufferView<float3> &positions,
                     const BufferView<TriangleHandle> &indices,
                     const std::vector<MeshHandle> &meshes,
                     const BufferView<uint> &instances,
                     const BufferView<float4x4> &transforms,
                     bool is_static);
    
    ~CudaAcceleration() noexcept override = default;
};

}

#endif

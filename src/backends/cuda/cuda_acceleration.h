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

#include <optix.h>

namespace luisa::cuda {

struct alignas(8) Traversable {
    luisa::uint2 handle;
};

struct alignas(16) Instance {
    std::array<luisa::float4, 3> transform;
    uint instance_id;
    uint sbt_offset;
    uint mask;
    uint flags;
    Traversable traversable;
    luisa::uint2 pad;
};

static_assert(sizeof(Instance) == 80);

}

LUISA_STRUCT(luisa::cuda::Traversable, handle)
LUISA_STRUCT(luisa::cuda::Instance, transform, instance_id, sbt_offset, mask, flags, traversable, pad)

namespace luisa::cuda {

using compute::Acceleration;
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
    OptixDeviceContext _optix_ctx{nullptr};
    OptixTraversableHandle _ias_handle{0u};
    BufferView<uchar> _ias_buffer;
    BufferView<Instance> _instance_buffer;
    BufferView<float4x4> _instance_transform_buffer;
    BufferView<uchar> _instance_update_buffer;
    KernelView _instance_update_kernel;
    BufferView<Traversable> _gas_handle_buffer;
    std::vector<OptixTraversableHandle> _gas_handles;
    std::vector<BufferView<uchar>> _gas_buffers;

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

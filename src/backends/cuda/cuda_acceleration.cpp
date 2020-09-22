//
// Created by Mike on 9/20/2020.
//

#ifdef LUISA_OPTIX_AVAILABLE

#include "cuda_acceleration.h"
#include "cuda_device.h"

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
    bool is_static) : _device{device} {
    
    
}

void CudaAcceleration::_refit(Dispatcher &dispatch) {

}

void CudaAcceleration::_intersect_any(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<AnyHit> &hit_buffer) const {

}

void CudaAcceleration::_intersect_closest(Dispatcher &dispatch, const BufferView<Ray> &ray_buffer, const BufferView<ClosestHit> &hit_buffer) const {

}

}

#endif

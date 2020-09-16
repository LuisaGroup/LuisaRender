//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/buffer.h>
#include <compute/device.h>
#include "material_handle.h"

namespace luisa::render {

using compute::Device;
using compute::BufferView;

struct InteractionBuffers {
    
    BufferView<bool> valid;
    BufferView<float3> pi;
    
    // Note:
    // these vectors should not be normalized,
    // since their lengths indicate distances.
    BufferView<float3> ray_origin_to_hit;
    
    BufferView<float3> ng;
    BufferView<float3> ns;
    BufferView<float2> uv;
    BufferView<MaterialHandle> material;
    
    void create(Device *device, size_t size) noexcept {
        valid = device->allocate_buffer<bool>(size);
        pi = device->allocate_buffer<float3>(size);
        ray_origin_to_hit = device->allocate_buffer<float3>(size);
        ng = device->allocate_buffer<float3>(size);
        ns = device->allocate_buffer<float3>(size);
        uv = device->allocate_buffer<float2>(size);
        material = device->allocate_buffer<MaterialHandle>(size);
    }
};

}

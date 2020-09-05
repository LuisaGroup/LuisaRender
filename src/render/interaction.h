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

struct Interaction {
    float3 pi;
    float3 wo;
    float3 ng;
    float3 ns;
    float2 uv;
    MaterialHandle material;
};

struct InteractionBuffers {
    
    BufferView<float3> pi;
    BufferView<float3> wo;
    BufferView<float3> ng;
    BufferView<float3> ns;
    BufferView<float2> uv;
    BufferView<MaterialHandle> material;
    
    InteractionBuffers(Device *device, size_t size) noexcept
        : pi{device->allocate_buffer<float3>(size)},
          wo{device->allocate_buffer<float3>(size)},
          ng{device->allocate_buffer<float3>(size)},
          ns{device->allocate_buffer<float3>(size)},
          uv{device->allocate_buffer<float2>(size)},
          material{device->allocate_buffer<MaterialHandle>(size)} {}
    
};

}

LUISA_STRUCT(luisa::render::Interaction, pi, wo, ng, ns, uv, material)

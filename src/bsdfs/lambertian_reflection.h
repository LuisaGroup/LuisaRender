//
// Created by Mike Smith on 2020/3/8.
//

#pragma once

#include <core/bsdf.h>

namespace luisa::bsdf::lambertian_reflection {

struct Data {
    packed_float3 albedo;
    float scale;
};

LUISA_DEVICE_CALLABLE void evaluate(
    LUISA_DEVICE_SPACE const Data *data_buffer,
    LUISA_DEVICE_SPACE const bsdf::Selection *queue,
    uint queue_size,
    LUISA_DEVICE_SPACE float4 *scattering_and_pdf_buffer,
    LUISA_DEVICE_SPACE float4 *wi_and_pdf_buffer,
    uint tid) {
    
    if (tid < queue_size) {
        
        auto selection = queue[tid];
        
        // sample if should
        if (selection.should_sample_wi()) {
        
        }
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

namespace luisa {

class LambertianReflection {
    
};

}

#endif

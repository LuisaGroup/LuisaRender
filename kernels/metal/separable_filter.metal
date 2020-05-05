//
// Created by Mike Smith on 2020/5/1.
//

#include "compatibility.h"
#include <core/filter.h>

LUISA_KERNEL void importance_sample_pixels(
    LUISA_DEVICE_SPACE const luisa::float2 *random_buffer,
    LUISA_DEVICE_SPACE luisa::float2 *pixel_location_buffer,
    LUISA_DEVICE_SPACE luisa::float3 *pixel_weight_buffer,
    LUISA_UNIFORM_SPACE luisa::filter::separable::LUT &lut,
    LUISA_UNIFORM_SPACE luisa::filter::separable::ImportanceSamplePixelsKernelUniforms &uniforms,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::filter::separable::importance_sample_pixels(random_buffer, pixel_location_buffer, pixel_weight_buffer, lut, uniforms, tid.x);
}

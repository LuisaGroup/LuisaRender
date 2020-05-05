//
// Created by Mike Smith on 2020/5/1.
//

#include "compatibility.h"
#include <core/camera.h>

using namespace luisa;

LUISA_KERNEL void generate_pixel_samples_without_filter(
    LUISA_DEVICE_SPACE const float2 *sample_buffer,
    LUISA_DEVICE_SPACE float2 *pixel_buffer,
    LUISA_DEVICE_SPACE float3 *throughput_buffer,
    LUISA_UNIFORM_SPACE camera::GeneratePixelSamplesWithoutFilterKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    camera::generate_pixel_samples_without_filter(sample_buffer, pixel_buffer, throughput_buffer, uniforms, tid.x);
}

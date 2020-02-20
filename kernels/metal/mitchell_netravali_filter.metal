#include "compatibility.h"
#include <filters/mitchell_netravali_filter.h>

LUISA_KERNEL void mitchell_netravali_filter_apply_and_accumulate(
    LUISA_DEVICE_SPACE const luisa::float3 *ray_color_buffer,
    LUISA_DEVICE_SPACE const luisa::float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE luisa::float4 *accumulation_buffer,
    LUISA_UNIFORM_SPACE luisa::filter::mitchell_netravali::ApplyAndAccumulateKernelUniforms &uniforms,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::filter::mitchell_netravali::apply_and_accumulate(ray_color_buffer, ray_pixel_buffer, accumulation_buffer, uniforms, tid.x);
}
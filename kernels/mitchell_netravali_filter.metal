#include "compatibility.h"
#include <filters/mitchell_netravali_filter.h>

using namespace luisa;

LUISA_KERNEL void mitchell_netravali_filter_apply(
    LUISA_DEVICE_SPACE const float3 *ray_radiance_buffer,
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE float4 *frame,
    LUISA_UNIFORM_SPACE mitchell_netravali_filter::ApplyKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    mitchell_netravali_filter::apply(ray_radiance_buffer, ray_pixel_buffer, frame, uniforms, tid);
}

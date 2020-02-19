#include "compatibility.h"
#include <cameras/thin_lens_camera.h>

using namespace luisa;

LUISA_KERNEL void thin_lens_camera_generate_rays(
    LUISA_DEVICE_SPACE const float4 *sample_buffer,
    LUISA_DEVICE_SPACE float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_DEVICE_SPACE float3 *ray_throughput_buffer,
    LUISA_UNIFORM_SPACE camera::thin_lens::GenerateRaysKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) noexcept {
    
    camera::thin_lens::generate_rays(sample_buffer, ray_pixel_buffer, ray_buffer, ray_throughput_buffer, uniforms, tid.x);
}

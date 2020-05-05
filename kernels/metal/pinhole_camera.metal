#include "compatibility.h"
#include <cameras/pinhole_camera.h>

using namespace luisa;

LUISA_KERNEL void generate_rays(
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_UNIFORM_SPACE camera::pinhole::GenerateRaysKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    camera::pinhole::generate_rays(ray_pixel_buffer, ray_buffer, uniforms, tid.x);
}

#include "compatibility.h"

#include <core/data_types.h>
#include <core/mathematics.h>
#include <core/ray.h>
#include <cameras/thin_lens_camera.h>

using namespace luisa;

LUISA_KERNEL void thin_lens_camera_generate_rays(
    LUISA_DEVICE_SPACE float3 *ray_throughput_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_queue_size,
    LUISA_UNIFORM_SPACE ThinLensCameraGenerateRaysKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_queue_size) {
        
        auto ray_index = ray_queue[tid.x];
        auto pixel = ray_pixel_buffer[ray_index];
        
    }
    
}

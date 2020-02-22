#include "compatibility.h"
#include <core/illumination.h>

using namespace luisa;

LUISA_KERNEL void illumination_uniform_select_lights(
    LUISA_DEVICE_SPACE const float *sample_buffer,
    LUISA_DEVICE_SPACE const illumination::Info *info_buffer,
    LUISA_DEVICE_SPACE Atomic<uint> *queue_sizes,
    LUISA_DEVICE_SPACE light::Selection *queues,
    LUISA_DEVICE_SPACE const uint &ray_count,
    LUISA_UNIFORM_SPACE illumination::SelectLightsKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    illumination::uniform_select_lights(sample_buffer, info_buffer, queue_sizes, queues, ray_count, uniforms, tid.x);
}
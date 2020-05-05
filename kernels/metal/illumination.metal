#include "compatibility.h"
#include <core/illumination.h>

using namespace luisa;

LUISA_KERNEL void uniform_select_lights(
    LUISA_DEVICE_SPACE const float *sample_buffer,
    LUISA_DEVICE_SPACE const illumination::Info *info_buffer,
    LUISA_DEVICE_SPACE Atomic<uint> *queue_sizes,
    LUISA_DEVICE_SPACE light::Selection *queues,
    LUISA_DEVICE_SPACE const uint &its_count,
    LUISA_UNIFORM_SPACE illumination::SelectLightsKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    illumination::uniform_select_lights(sample_buffer, info_buffer, queue_sizes, queues, its_count, uniforms, tid.x);
}

LUISA_KERNEL void collect_light_interactions(
    LUISA_DEVICE_SPACE const uint *its_instance_id_buffer,
    LUISA_DEVICE_SPACE const uint8_t *its_state_buffer,
    LUISA_DEVICE_SPACE const illumination::Info *instance_to_info_buffer,
    LUISA_DEVICE_SPACE Atomic<uint> *queue_sizes,
    LUISA_DEVICE_SPACE light::Selection *queues,
    LUISA_DEVICE_SPACE const uint &its_count,
    LUISA_UNIFORM_SPACE illumination::CollectLightInteractionsKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    illumination::collect_light_interactions(its_instance_id_buffer, its_state_buffer, instance_to_info_buffer, queue_sizes, queues, its_count, uniforms, tid.x);
}

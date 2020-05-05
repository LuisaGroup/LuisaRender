#include "compatibility.h"
#include <lights/point.h>

using namespace luisa;

LUISA_KERNEL void generate_samples(
    LUISA_DEVICE_SPACE const light::point::Data *data_buffer,
    LUISA_DEVICE_SPACE const light::Selection *queue,
    LUISA_DEVICE_SPACE const uint &queue_size,
    LUISA_DEVICE_SPACE uint8_t *its_state_buffer,
    LUISA_DEVICE_SPACE const float3 *its_position_buffer,
    LUISA_DEVICE_SPACE float4 *Li_and_pdf_w_buffer,
    LUISA_DEVICE_SPACE bool *is_delta_buffer,
    LUISA_DEVICE_SPACE Ray *shadow_ray_buffer,
    uint2 tid [[thread_position_in_grid]]) {
    
    light::point::generate_samples(data_buffer, queue, queue_size, its_state_buffer, its_position_buffer, Li_and_pdf_w_buffer, is_delta_buffer, shadow_ray_buffer, tid.x);
}

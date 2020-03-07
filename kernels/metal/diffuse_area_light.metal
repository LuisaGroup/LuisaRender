#include "compatibility.h"
#include <lights/diffuse_area_light.h>

using namespace luisa;

LUISA_KERNEL void diffuse_area_light_generate_samples(
    LUISA_DEVICE_SPACE const light::diffuse_area::Data *data_buffer,
    LUISA_DEVICE_SPACE const float3 *sample_buffer,
    LUISA_DEVICE_SPACE const float4x4 *transform_buffer,
    LUISA_DEVICE_SPACE const float *cdf_buffer,
    LUISA_DEVICE_SPACE const packed_uint3 *index_buffer,
    LUISA_DEVICE_SPACE const float3 *position_buffer,
    LUISA_DEVICE_SPACE const float3 *normal_buffer,
    LUISA_DEVICE_SPACE const light::Selection *queue,
    LUISA_DEVICE_SPACE const uint &queue_size,
    LUISA_DEVICE_SPACE const uint8_t *its_state_buffer,
    LUISA_DEVICE_SPACE const float3 *its_position_buffer,
    LUISA_DEVICE_SPACE float4 *Li_and_pdf_w_buffer,
    LUISA_DEVICE_SPACE Ray *shadow_ray_buffer,
    uint2 tid [[thread_position_in_grid]]) {
    
    light::diffuse_area::generate_samples(
        data_buffer, sample_buffer, transform_buffer, cdf_buffer,
        index_buffer, position_buffer, normal_buffer,
        queue, queue_size,
        its_state_buffer, its_position_buffer,
        Li_and_pdf_w_buffer, shadow_ray_buffer,
        tid.x);
}

LUISA_KERNEL void diffuse_area_light_evaluate_emissions(
    LUISA_DEVICE_SPACE const light::diffuse_area::Data *data_buffer,
    LUISA_DEVICE_SPACE const light::Selection *queue,
    LUISA_DEVICE_SPACE const uint &queue_size,
    LUISA_DEVICE_SPACE uint8_t *its_state_buffer,
    LUISA_DEVICE_SPACE float3 *its_emission_buffer,
    uint2 tid [[thread_position_in_grid]]) {
    
    light::diffuse_area::evaluate_emissions(data_buffer, queue, queue_size, its_state_buffer, its_emission_buffer, tid.x);
}

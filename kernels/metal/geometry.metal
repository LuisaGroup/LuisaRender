#include "compatibility.h"
#include <core/geometry.h>

using namespace luisa;

LUISA_KERNEL void geometry_evaluate_interactions(
    LUISA_DEVICE_SPACE const uint &ray_count,
    LUISA_DEVICE_SPACE const Ray *ray_buffer,
    LUISA_DEVICE_SPACE const ClosestHit *hit_buffer,
    LUISA_DEVICE_SPACE const float3 *position_buffer,
    LUISA_DEVICE_SPACE const float3 *normal_buffer,
    LUISA_DEVICE_SPACE const float2 *uv_buffer,
    LUISA_DEVICE_SPACE const packed_uint3 *index_buffer,
    LUISA_DEVICE_SPACE const uint *vertex_offset_buffer,
    LUISA_DEVICE_SPACE const uint *index_offset_buffer,
    LUISA_DEVICE_SPACE const float4x4 *transform_buffer,
    LUISA_DEVICE_SPACE uint8_t *interaction_state_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_position_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_normal_buffer,
    LUISA_DEVICE_SPACE float2 *interaction_uv_buffer,
    LUISA_DEVICE_SPACE float4 *interaction_wo_and_distance_buffer,
    LUISA_UNIFORM_SPACE geometry::EvaluateInteractionsKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    geometry::evaluate_interactions(
        ray_count, ray_buffer, hit_buffer, position_buffer, normal_buffer, uv_buffer,
        index_buffer, vertex_offset_buffer, index_offset_buffer, transform_buffer,
        interaction_state_buffer, interaction_position_buffer, interaction_normal_buffer, interaction_uv_buffer, interaction_wo_and_distance_buffer,
        uniforms, tid.x);
}
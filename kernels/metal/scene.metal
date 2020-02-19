#include "compatibility.h"
#include <core/scene.h>

using namespace luisa;

LUISA_KERNEL void scene_evaluate_interactions(
    LUISA_DEVICE_SPACE const uint &ray_count,
    LUISA_DEVICE_SPACE const Ray *ray_buffer,
    LUISA_DEVICE_SPACE const ClosestHit *hit_buffer,
    LUISA_DEVICE_SPACE const float3 *position_buffer,
    LUISA_DEVICE_SPACE const float3 *normal_buffer,
    LUISA_DEVICE_SPACE const float2 *uv_buffer,
    LUISA_DEVICE_SPACE const packed_uint3 *index_buffer,
    LUISA_DEVICE_SPACE const float4x4 *transform_buffer,
    LUISA_DEVICE_SPACE const MaterialInfo *material_info_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_position_buffer,
    LUISA_DEVICE_SPACE float3 *interaction_normal_buffer,
    LUISA_DEVICE_SPACE float2 *interaction_uv_buffer,
    LUISA_DEVICE_SPACE float4 *interaction_wo_and_distance_buffer,
    LUISA_DEVICE_SPACE MaterialInfo *interaction_material_info_buffer,
    LUISA_UNIFORM_SPACE uint &attribute_flags,
    uint2 tid [[thread_position_in_grid]]) {
    
    scene::evaluate_interactions(
        ray_count, ray_buffer, hit_buffer,
        position_buffer, normal_buffer, uv_buffer, index_buffer, transform_buffer, material_info_buffer,
        interaction_position_buffer, interaction_normal_buffer, interaction_uv_buffer, interaction_wo_and_distance_buffer, interaction_material_info_buffer,
        attribute_flags, tid.x);
}
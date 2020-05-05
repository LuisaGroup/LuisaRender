#include "compatibility.h"
#include <core/film.h>

using namespace luisa;

LUISA_KERNEL void reset_accumulation_buffer(
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    film::reset_accumulation_buffer(accumulation_buffer, pixel_count, tid.x);
}

LUISA_KERNEL void accumulate_tile(
    LUISA_DEVICE_SPACE const float3 *ray_color_buffer,
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    LUISA_UNIFORM_SPACE film::AccumulateTileKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    film::accumulate_tile(ray_color_buffer, accumulation_buffer, uniforms, tid.x);
}

//
// Created by Mike Smith on 2019/10/21.
//

#include "compatibility.h"
#include <core/film.h>

using namespace luisa;

LUISA_KERNEL void film_clear_accumulation_buffer(
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    film::clear_accumulation_buffer(accumulation_buffer, pixel_count, tid.x);
}

LUISA_KERNEL void film_accumulate_frame(
    LUISA_DEVICE_SPACE const float4 *frame,
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    film::accumulate_frame(frame, accumulation_buffer, pixel_count, tid.x);
}

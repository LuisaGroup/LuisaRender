//
// Created by Mike Smith on 2019/10/21.
//

#include "compatibility.h"

#include <core/data_types.h>

using namespace luisa;

LUISA_KERNEL void film_clear_accumulation_buffer(
    LUISA_DEVICE_SPACE int4 *accumulation_buffer,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < pixel_count) {
        accumulation_buffer[tid.x] = {};
    }
    
}

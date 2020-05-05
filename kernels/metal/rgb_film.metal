#include "compatibility.h"
#include <films/rgb.h>

using namespace luisa;

LUISA_KERNEL void postprocess(
    LUISA_DEVICE_SPACE float4 *accumulation_buffer,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    film::rgb::postprocess(accumulation_buffer, pixel_count, tid.x);
}

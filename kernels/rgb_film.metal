#include "compatibility.h"

#include <core/color_spaces.h>

using namespace luisa;

LUISA_KERNEL void rgb_film_postprocess(
    LUISA_DEVICE_SPACE const int4 *accumulation_buffer,
    LUISA_DEVICE_SPACE packed_float3 *framebuffer,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < pixel_count) {
        auto index = tid.x;
        auto f = make_float4(accumulation_buffer[index]);
        framebuffer[index] = make_packed_float3(XYZ2RGB(make_float3(f) / f.a));
    }
}
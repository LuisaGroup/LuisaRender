//
// Created by Mike Smith on 2019/10/21.
//

#include "compatibility.h"

#include <core/data_types.h>
#include <core/mathematics.h>

#include <color_spaces.h>
#include <frame_data.h>
#include <ray_data.h>

using namespace luisa;
using namespace luisa::math;

inline float Mitchell1D(float x) {
    constexpr auto B = 1.0f / 3.0f;
    constexpr auto C = 1.0f / 3.0f;
    x = min(abs(2 * x), 2.0f);
    auto xx = x * x;
    
    return (1.0f / 6.0f) *
           (x > 1 ?
            ((-B - 6 * C) * xx + (6 * B + 30 * C) * x + (-12 * B - 48 * C)) * x + (8 * B + 24 * C) :
            ((12 - 9 * B - 6 * C) * xx + (-18 + 12 * B + 6 * C) * x) * x + (6 - 2 * B));
}

LUISA_KERNEL void rgb_film_clear(
    LUISA_DEVICE_SPACE int4 *accum_buffer,
    LUISA_PRIVATE_SPACE uint &ray_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_count) {
        accum_buffer[tid.x] = {};
    }
    
}

LUISA_KERNEL void rgb_film_gather_rays(
    LUISA_DEVICE_SPACE const GatherRayData *ray_buffer,
    LUISA_PRIVATE_SPACE FrameData &frame_data,
    LUISA_DEVICE_SPACE Atomic<int> *accum_buffer,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        auto index = tid.x + tid.y * frame_data.size.x;
        auto new_value = int4(int3(round(ACEScg2XYZ(ray_buffer[index].radiance) * 1024.0f)), 1);
        luisa_atomic_fetch_add(accum_buffer[index * 4u + 0u], new_value.x);
        luisa_atomic_fetch_add(accum_buffer[index * 4u + 1u], new_value.y);
        luisa_atomic_fetch_add(accum_buffer[index * 4u + 2u], new_value.z);
        luisa_atomic_fetch_add(accum_buffer[index * 4u + 3u], new_value.w);
    }
}

LUISA_KERNEL void rgb_film_convert_colorspace(
    LUISA_PRIVATE_SPACE FrameData &frame_data,
    LUISA_DEVICE_SPACE const int4 *accum_buffer,
    LUISA_DEVICE_SPACE packed_float3 *result_buffer,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        auto index = tid.x + tid.y * frame_data.size.x;
        auto f = make_float4(accum_buffer[index]);
        result_buffer[index] = make_packed_float3(XYZ2RGB(make_float3(f) / (1024.0f * f.a)));
    }
}

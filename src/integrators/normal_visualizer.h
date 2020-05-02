//
// Created by Mike Smith on 2020/2/19.
//

#pragma once

#include <core/data_types.h>
#include <core/colorspaces.h>
#include <core/interaction.h>

namespace luisa::integrator::normal {

LUISA_DEVICE_CALLABLE inline void colorize_normals(
    LUISA_DEVICE_SPACE float3 *normals,
    LUISA_DEVICE_SPACE const float3 *throughput_buffer,
    LUISA_DEVICE_SPACE const uint8_t *state_buffer,
    uint pixel_count,
    uint tid) noexcept {
    
    if (tid < pixel_count) {
        auto n = (state_buffer[tid] & interaction::state::HIT) != 0u ? normals[tid] : make_float3(-1.0f);
        auto w = throughput_buffer[tid];
        normals[tid] = w * XYZ2ACEScg(RGB2XYZ(n * 0.5f + 0.5f));
    }
}

}

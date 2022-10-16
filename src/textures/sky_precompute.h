//
// Created by Mike Smith on 2022/10/13.
//

#pragma once

#include <core/basic_types.h>

namespace luisa::render {

struct NishitaSkyData {
    float sun_elevation;
    float sun_angle;
    float altitude;
    float air_density;
    float dust_density;
    float ozone_density;
};

struct NishitaSkyPrecomputedSun {
    float3 bottom;
    float3 top;
};

void SKY_nishita_skymodel_precompute_texture(
    NishitaSkyData data, float4 *pixels,
    uint2 resolution, uint2 y_range) noexcept;

[[nodiscard]] NishitaSkyPrecomputedSun
SKY_nishita_skymodel_precompute_sun(
    NishitaSkyData data) noexcept;

}// namespace luisa::render

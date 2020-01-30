//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

LUISA_DEVICE_CALLABLE inline float RGB2Luminance(float3 rgb) {
    return 0.2126729f * rgb.r + 0.7151522f * rgb.g + 0.0721750f * rgb.b;
}

LUISA_DEVICE_CALLABLE inline float XYZ2Luminance(float3 xyz) {
    return xyz.y;
}

LUISA_DEVICE_CALLABLE inline float ACES2Luminance(float3 aces) {
    return 0.3439664498f * aces.r + 0.7281660966f * aces.g - 0.0721325464f * aces.b;
}

LUISA_DEVICE_CALLABLE inline float ACEScg2Luminance(float3 cg) {
    return 0.2722287168f * cg.r + 0.6740817658f * cg.g + 0.0536895174f * cg.b;
}

LUISA_DEVICE_CALLABLE inline float3 XYZ2RGB(float3 xyz) {
    return {3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
            -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
            0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z};
}

LUISA_DEVICE_CALLABLE inline float3 RGB2XYZ(float3 rgb) {
    return {0.4124564f * rgb.r + 0.3575761f * rgb.g + 0.1804375f * rgb.b,
            0.2126729f * rgb.r + 0.7151522f * rgb.g + 0.0721750f * rgb.b,
            0.0193339f * rgb.r + 0.1191920f * rgb.g + 0.9503041f * rgb.b};
}

LUISA_DEVICE_CALLABLE inline float3 ACES2XYZ(float3 aces) {
    return {0.9525523959f * aces.r + 0.0000000000f * aces.g + 0.0000936786f * aces.b,
            0.3439664498f * aces.r + 0.7281660966f * aces.g - 0.0721325464f * aces.b,
            0.0000000000f * aces.r + 0.0000000000f * aces.g + 1.0088251844f * aces.b};
}

LUISA_DEVICE_CALLABLE inline float3 XYZ2ACES(float3 xyz) {
    return {1.0498110175f * xyz.x + 0.0000000000f * xyz.y - 0.0000974845f * xyz.z,
            -0.4959030231f * xyz.x + 1.3733130458f * xyz.y + 0.0982400361f * xyz.z,
            0.0000000000f * xyz.x + 0.0000000000f * xyz.y + 0.9912520182f * xyz.z};
}

LUISA_DEVICE_CALLABLE inline float3 ACEScg2XYZ(float3 cg) {
    return {0.6624541811f * cg.r + 0.1340042065f * cg.g + 0.1561876870f * cg.b,
            0.2722287168f * cg.r + 0.6740817658f * cg.g + 0.0536895174f * cg.b,
            -0.0055746495f * cg.r + 0.0040607335f * cg.g + 1.0103391003f * cg.b};
}

LUISA_DEVICE_CALLABLE inline float3 XYZ2ACEScg(float3 xyz) {
    return {1.6410233797f * xyz.x - 0.3248032942 * xyz.y - 0.2364246952 * xyz.z,
            -0.6636628587f * xyz.x + 1.6153315917 * xyz.y + 0.0167563477 * xyz.z,
            0.0117218943f * xyz.x - 0.0082844420 * xyz.y + 0.9883948585 * xyz.z};
}
    
}
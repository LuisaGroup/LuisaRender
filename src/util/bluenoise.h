// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using compute::Expr;
using compute::Volume;

static constexpr auto BlueNoiseResolution = 128u;
static constexpr auto NumBlueNoiseTextures = 48u;

extern const uint16_t BlueNoiseTextures[NumBlueNoiseTextures][BlueNoiseResolution][BlueNoiseResolution];

// Blue noise lookup functions
inline auto BlueNoise(Expr<Volume<float>> textures, Expr<uint> textureIndex, Expr<uint2> p) noexcept {
    auto t = textureIndex % NumBlueNoiseTextures;
    auto x = p.x % BlueNoiseResolution;
    auto y = p.y % BlueNoiseResolution;
    return textures.read(make_uint3(y, x, t)).x;
}

}// namespace luisa::render

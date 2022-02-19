// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#pragma once

#include <core/platform.h>

namespace luisa::render {

static constexpr auto BlueNoiseResolution = 128u;
static constexpr auto NumBlueNoiseTextures = 48u;

#ifndef LUISA_RENDER_BLUE_NOISE_DEFINITION
LUISA_IMPORT_API const uint16_t BlueNoiseTextures[NumBlueNoiseTextures][BlueNoiseResolution][BlueNoiseResolution];
#endif

}// namespace luisa::render

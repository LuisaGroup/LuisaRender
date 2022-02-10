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

}// namespace luisa::render

// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#pragma once

#include <core/platform.h>

namespace luisa::render {

static constexpr auto nPMJ02bnSets = 5u;
static constexpr auto nPMJ02bnSamples = 65536u;

#ifndef LUISA_RENDER_PMJ02_TABLES_DEFINITION
LUISA_IMPORT_API const uint32_t PMJ02bnSamples[nPMJ02bnSets][nPMJ02bnSamples][2];
#endif

}// namespace luisa::render

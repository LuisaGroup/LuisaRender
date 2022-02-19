// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#pragma once

#include <core/platform.h>

namespace luisa::render {

// Sobol Matrix Declarations
static constexpr auto NSobolDimensions = 1024u;
static constexpr auto SobolMatrixSize = 52u;

static constexpr auto VdCSobolMatrixSize = 25u;
static constexpr auto VdCSobolMatrixInvSize = 26u;

#ifndef LUISA_RENDER_SOBOL_MATRICES_DEFINITION
LUISA_IMPORT_API const uint32_t SobolMatrices32[NSobolDimensions * SobolMatrixSize];
LUISA_IMPORT_API const uint64_t VdCSobolMatrices[VdCSobolMatrixSize][SobolMatrixSize];
LUISA_IMPORT_API const uint64_t VdCSobolMatricesInv[VdCSobolMatrixInvSize][SobolMatrixSize];
#endif

}// namespace luisa::render

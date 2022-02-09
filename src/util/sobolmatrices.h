// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#pragma once

#include <cstdint>

namespace luisa::render {

// Sobol Matrix Declarations
static constexpr auto NSobolDimensions = 1024u;
static constexpr auto SobolMatrixSize = 52u;
extern const uint32_t SobolMatrices32[NSobolDimensions * SobolMatrixSize];
extern const uint64_t VdCSobolMatrices[][SobolMatrixSize];
extern const uint64_t VdCSobolMatricesInv[][SobolMatrixSize];

}  // namespace pbrt

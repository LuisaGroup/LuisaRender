//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "compatibility.h"

using BRDFTag = uint32_t;

struct BRDFData {
    uint8_t data[256];  // should be large enough even for complex BRDFs like Disney BRDF
};

static_assert(sizeof(BRDFData) == 256ul);

struct MaterialData {  // for now, only lambert is supported
    PackedVec3f albedo;
    uint is_mirror;
};

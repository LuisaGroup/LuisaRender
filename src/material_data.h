//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "../src/compatibility.h"

struct MaterialData {  // for now, only lambert is supported
    PackedVec3f albedo;
    uint is_mirror;
};

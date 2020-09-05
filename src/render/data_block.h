//
// Created by Mike Smith on 2020/9/6.
//

#pragma once

#include <compute/type_desc.h>

namespace luisa::render {

struct DataBlock {
    uint4 padding;
};

}

LUISA_STRUCT(luisa::render::DataBlock, padding)

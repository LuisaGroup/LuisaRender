//
// Created by Mike Smith on 2020/9/29.
//

#pragma once

#include <compute/dsl.h>

namespace luisa::render {

struct alignas(16) DataBlock {
    float4 padding;
};

}

LUISA_STRUCT(luisa::render::DataBlock, padding)

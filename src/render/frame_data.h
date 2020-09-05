//
// Created by Mike Smith on 2020/9/6.
//

#pragma once

#include <compute/type_desc.h>

namespace luisa::render {

struct FrameData {
    uint index;
    float time;
    uint2 offset;
    uint2 size;
};

}

LUISA_STRUCT(luisa::render::FrameData, index, time, offset, size)

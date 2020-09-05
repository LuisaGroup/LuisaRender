//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/type_desc.h>

namespace luisa::render {

struct MaterialHandle {
    int type_id;
    uint data_offset;
};

}

LUISA_STRUCT(luisa::render::MaterialHandle, type_id, data_offset)

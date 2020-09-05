//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/type_desc.h>
#include <compute/buffer.h>

#include "plugin.h"

namespace luisa::render {

struct MaterialHandle {
    int type_id;
    uint data_block_offset;
};

struct MaterialDataBlock {
    uint4 padding;
};

}

LUISA_STRUCT(luisa::render::MaterialHandle, type_id, data_block_offset)
LUISA_STRUCT(luisa::render::MaterialDataBlock, padding)

namespace luisa::render {

class Material : public Plugin {

};

}

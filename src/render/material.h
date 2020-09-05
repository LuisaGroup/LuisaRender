//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/type_desc.h>
#include <compute/buffer.h>

#include "plugin.h"
#include "illumination.h"
#include "material_handle.h"

namespace luisa::render {

class Material : public Plugin, public Illumination {

public:
    struct DataBlock {
        uint4 padding;
    };

public:


};

}

LUISA_STRUCT(luisa::render::Material::DataBlock, padding)

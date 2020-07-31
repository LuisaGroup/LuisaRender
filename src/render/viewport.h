//
// Created by Mike Smith on 2020/2/18.
//

#pragma once

#include <compute/data_types.h>
#include <compute/type_desc.h>

namespace luisa {

struct Viewport {
    uint2 origin;
    uint2 size;
};

#ifndef LUISA_DEVICE_COMPATIBLE

namespace dsl {
LUISA_STRUCT(Viewport, origin, size)
}

#endif

}

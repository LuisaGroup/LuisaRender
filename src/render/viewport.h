//
// Created by Mike Smith on 2020/2/18.
//

#pragma once

#include <core/data_types.h>
#include <compute/type_desc.h>

namespace luisa {

struct Viewport {
    uint2 origin;
    uint2 size;
};

#ifndef LUISA_DEVICE_COMPATIBLE

namespace compute::dsl {
LUISA_STRUCT(Viewport, origin, size)
}

#endif

}

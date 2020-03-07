//
// Created by Mike Smith on 2020/3/7.
//

#pragma once

#include "data_types.h"

namespace luisa::appearance {

static constexpr auto MAX_BSDF_LAYER_COUNT = 8u;

struct Info {
    uint16_t weights[MAX_BSDF_LAYER_COUNT];  // weights of BSDF layers, scaled by 65535
    uint8_t tags[MAX_BSDF_LAYER_COUNT];      // tags of BSDF layers
    uint32_t indices[MAX_BSDF_LAYER_COUNT];  // indices of data of BSDF layers, counting from zero individually for each BSDF type
};

static_assert(sizeof(Info) == (sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint32_t)) * MAX_BSDF_LAYER_COUNT);

}

#ifndef LUISA_DEVICE_COMPATIBLE

namespace luisa {

class Appearance {

};

}

#endif

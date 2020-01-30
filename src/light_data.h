//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct LightData {  // for now, only point light is supported
    float3 position;
    float3 emission;
};

struct LightSample {
    packed_float3 radiance;
    float pdf;
    packed_float3 direction;
    float distance;
};

}

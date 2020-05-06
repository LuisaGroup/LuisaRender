//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include <core/ray.h>
#include <core/interaction.h>
#include <core/mathematics.h>
#include <core/light.h>

namespace luisa::light::point {

struct Data {
    float3 position;
    float3 emission;
};

}

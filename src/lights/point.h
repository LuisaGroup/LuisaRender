//
// Created by Mike Smith on 2020/2/17.
//

#pragma once

#include <render/ray.h>
#include <render/interaction.h>
#include <compute/mathematics.h>
#include <render/light.h>

namespace luisa::light::point {

struct Data {
    float3 position;
    float3 emission;
};

}

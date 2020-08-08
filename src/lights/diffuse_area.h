//
// Created by Mike Smith on 2020/2/22.
//

#pragma once

#include <core/mathematics.h>
#include <render/ray.h>
#include <render/interaction.h>
#include <render/sampling.h>
#include <render/light.h>

namespace luisa::light::diffuse_area {

struct Data {
    float3 emission;
    uint2 cdf_range;
    uint instance_id;
    uint triangle_offset;
    uint vertex_offset;
    float shape_area;
    bool two_sided;
};

static_assert(sizeof(Data) == 48ul);

}

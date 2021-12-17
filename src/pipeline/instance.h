//
// Created by Mike on 2021/12/17.
//

#pragma once

#include <luisa-compute.h>

namespace luisa::render {

struct alignas(16) Instance {

    // vertex buffer information
    uint position_buffer_id;
    uint normal_buffer_id;
    uint tangent_buffer_id;
    uint uv_buffer_id;// 16B

    // index buffer information
    uint triangle_buffer_id;
    uint triangle_count;// 24B

    // transforming & sampling
    uint transform_buffer_id;
    uint area_cdf_buffer_id;// 32B

    // appearance & illumination
    uint material_tag;
    uint material_buffer_id;
    uint light_tag;
    uint light_buffer_id;// 48B
};

static_assert(sizeof(Instance) == 48);

}// namespace luisa::render

LUISA_STRUCT(luisa::render::Instance,

             position_buffer_id,
             normal_buffer_id,
             tangent_buffer_id,
             uv_buffer_id,

             triangle_buffer_id,
             triangle_count,
             transform_buffer_id,
             area_cdf_buffer_id,

             material_tag,
             material_buffer_id,
             light_tag,
             light_buffer_id){};

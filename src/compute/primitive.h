//
// Created by Mike Smith on 2020/9/11.
//

#pragma once

#include <compute/dsl.h>

namespace luisa::compute {

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

struct TriangleHandle {
    uint i;
    uint j;
    uint k;
};

struct EntityHandle {
    uint vertex_offset;
    uint triangle_offset;
};

struct EntityRange : public EntityHandle {
    uint triangle_count;
};

}

LUISA_STRUCT(luisa::compute::Vertex, position, normal, uv)
LUISA_STRUCT(luisa::compute::TriangleHandle, i, j, k)
LUISA_STRUCT(luisa::compute::EntityHandle, vertex_offset, triangle_offset)
LUISA_STRUCT(luisa::compute::EntityRange, vertex_offset, triangle_offset, triangle_count)

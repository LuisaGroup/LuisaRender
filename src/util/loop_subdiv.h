//
// Created by Mike Smith on 2022/11/8.
//

#pragma once

#include <core/stl.h>
#include <runtime/rtx/triangle.h>
#include <util/vertex.h>

namespace luisa::render {

using compute::Triangle;

struct SubdivMesh {
    luisa::vector<Vertex> vertices;
    luisa::vector<Triangle> triangles;
    luisa::vector<uint> base_triangle_indices;
};

[[nodiscard]] SubdivMesh loop_subdivide(luisa::span<const Vertex> vertices,
                                        luisa::span<const Triangle> triangles,
                                        uint level) noexcept;

}// namespace luisa::render

//
// Created by Mike Smith on 2022/11/8.
//

#pragma once

#include <core/stl.h>
#include <rtx/mesh.h>
#include <util/vertex.h>

namespace luisa::render {

using compute::Triangle;

struct alignas(16) SubdivVertex {
    float px;
    float py;
    float pz;
    uint n;
};

struct alignas(16) SubdivTriangle {
    Triangle t;
    uint base;
};

static_assert(sizeof(SubdivVertex) == 16u);
static_assert(sizeof(SubdivTriangle) == 16u);

struct SubdivMesh {
    std::vector<SubdivVertex> vertices;
    std::vector<SubdivTriangle> triangles;
};

[[nodiscard]] SubdivMesh loop_subdivide(luisa::span<const Vertex> vertices,
                                        luisa::span<const Triangle> triangles,
                                        uint level) noexcept;

}// namespace luisa::render

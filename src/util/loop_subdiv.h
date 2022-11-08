//
// Created by Mike Smith on 2022/11/8.
//

#pragma once

#include <core/stl.h>
#include <rtx/mesh.h>
#include <util/vertex.h>

namespace luisa::render {

using compute::Triangle;

[[nodiscard]] std::pair<luisa::vector<Vertex>, luisa::vector<Triangle>>
loop_subdivide(luisa::span<const Vertex> vertices,
               luisa::span<const Triangle> triangles,
               uint level) noexcept;

}// namespace luisa::render

//
// Created by Mike Smith on 2022/11/8.
//

#include <core/stl.h>
#include <rtx/mesh.h>

namespace luisa::render {

using compute::Triangle;

[[nodiscard]] std::pair<luisa::vector<float3>, luisa::vector<Triangle>>
loop_subdivide(luisa::span<const float3> positions,
               luisa::span<const Triangle> triangles,
               uint level) noexcept;

}// namespace luisa::render

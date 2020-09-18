//
// Created by Mike Smith on 2020/9/18.
//

#pragma once

#include <render/surface.h>

namespace luisa::render {
[[nodiscard]] std::unique_ptr<SurfaceShader> create_lambertian_reflection(float3 albedo, bool double_sided);
}

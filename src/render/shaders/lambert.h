//
// Created by Mike Smith on 2020/9/18.
//

#pragma once

#include <render/surface.h>

namespace luisa::render::shading {
[[nodiscard]] std::unique_ptr<SurfaceShader> create_lambert_reflection(float3 albedo, bool double_sided);
[[nodiscard]] std::unique_ptr<SurfaceShader> create_lambert_emission(float3 emission, bool double_sided);
}

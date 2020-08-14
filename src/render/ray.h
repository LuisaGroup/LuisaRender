//
// Created by Mike Smith on 2020/1/31.
//

#pragma once

#include <string>
#include <memory>
#include <string_view>
#include <map>

#include <core/data_types.h>
#include <core/mathematics.h>
#include <core/logging.h>

#include <compute/device.h>
#include <compute/buffer.h>
#include <compute/type_desc.h>

namespace luisa::render {

struct Ray {
    packed_float3 origin;
    float min_distance;
    packed_float3 direction;
    float max_distance;
};

}

// Register ray struct
LUISA_STRUCT(render::Ray, origin, min_distance, direction, max_distance)

namespace luisa::render {

inline auto make_ray(float3 o, float3 d, float t_min = 1e-4f, float t_max = INFINITY) noexcept {
    return Ray{make_packed_float3(o), t_min, make_packed_float3(d), t_max};
}

// Adapted from Ray Tracing Gems
inline float3 offset_ray_origin(float3 p, float3 n) noexcept {
    
    constexpr auto origin = 1.0f / 32.0f;
    constexpr auto float_scale = 1.0f / 65536.0f;
    constexpr auto int_scale = 256.0f;
    
    auto of_i = make_int3(static_cast<int>(int_scale * n.x), static_cast<int>(int_scale * n.y), static_cast<int>(int_scale * n.z));
    
    auto as_float = [](auto x) noexcept { return *reinterpret_cast<float *>(&x); };
    auto as_int = [](auto x) noexcept { return *reinterpret_cast<int *>(&x); };
    
    auto p_i = make_float3(
        as_float(as_int(p.x) + (p.x < 0 ? -of_i.x : of_i.x)),
        as_float(as_int(p.y) + (p.y < 0 ? -of_i.y : of_i.y)),
        as_float(as_int(p.z) + (p.z < 0 ? -of_i.z : of_i.z)));
    
    return make_float3(
        math::abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        math::abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        math::abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

}

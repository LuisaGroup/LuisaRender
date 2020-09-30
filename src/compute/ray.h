//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/dsl.h>

namespace luisa::compute {

struct alignas(16) Ray {
    float origin_x;
    float origin_y;
    float origin_z;
    float min_distance;
    float direction_x;
    float direction_y;
    float direction_z;
    float max_distance;
};

}

LUISA_STRUCT(luisa::compute::Ray, origin_x, origin_y, origin_z, min_distance, direction_x, direction_y, direction_z, max_distance)

namespace luisa::compute {

using dsl::Expr;
using dsl::Var;

inline Expr<float3> offset_ray_origin(Expr<float3> p_in, Expr<float3> n_in) noexcept {
    
    using namespace dsl;
    
    constexpr auto origin = 1.0f / 32.0f;
    constexpr auto float_scale = 1.0f / 65536.0f;
    constexpr auto int_scale = 256.0f;
    
    Var n = n_in;
    auto of_i = make_int3(cast<int>(int_scale * n.x), cast<int>(int_scale * n.y), cast<int>(int_scale * n.z));
    
    auto as_float = [](auto x) noexcept { return bitcast<float>(x); };
    auto as_int = [](auto x) noexcept { return bitcast<int>(x); };
    
    Var p = p_in;
    Var p_i = make_float3(
        as_float(as_int(p.x) + select(p.x < 0, -of_i.x, of_i.x)),
        as_float(as_int(p.y) + select(p.y < 0, -of_i.y, of_i.y)),
        as_float(as_int(p.z) + select(p.z < 0, -of_i.z, of_i.z)));
    
    return select(abs(p) < origin, p + float_scale * n, p_i);
}

}

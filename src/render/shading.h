//
// Created by Mike Smith on 2020/10/1.
//

#pragma once

#include <compute/dsl_syntax.h>

namespace luisa::render {

inline namespace shading {

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

}
}

LUISA_STRUCT(luisa::render::shading::Onb, tangent, binormal, normal)

namespace luisa::render {

inline namespace shading {

using compute::dsl::Var;
using compute::dsl::Expr;

inline Expr<Onb> make_onb(Var<float3> normal) noexcept {
    Var binormal = normalize(select(
        abs(normal.x) > abs(normal.z),
        make_float3(-normal.y, normal.x, 0.0f),
        make_float3(0.0f, -normal.z, normal.y)));
    Var tangent = normalize(cross(binormal, normal));
    Var<Onb> onb{tangent, binormal, normal};
    return onb;
}

inline Expr<float3> transform_to_local(Expr<Onb> onb, Var<float3> v) noexcept {
    return make_float3(dot(v, onb.tangent), dot(v, onb.binormal), dot(v, onb.normal));
}

inline Expr<float3> transform_to_world(Expr<Onb> onb, Var<float3> v) noexcept {
    return v.x * onb.tangent + v.y * onb.binormal + v.z * onb.normal;
}

inline Expr<float3> face_forward(Var<float3> d, Var<float3> ref) noexcept {
    return select(dot(d, ref) < 0.0f, -d, d);
}

inline Expr<float> sign(Var<float> x) noexcept {
    return select(x > 0.0f, 1.0f, -1.0f);
}

}
}

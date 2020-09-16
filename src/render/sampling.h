//
// Created by Mike Smith on 2020/9/16.
//

#pragma once

#include <compute/dsl.h>

namespace luisa::render {

inline namespace sampling {

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

}
}

LUISA_STRUCT(luisa::render::sampling::Onb, tangent, binormal, normal)

namespace luisa::render {

inline namespace sampling {

using compute::dsl::Var;
using compute::dsl::Expr;

inline Expr<Onb> make_onb(Expr<float3> normal) noexcept {
    Var binormal = normalize(select(
        abs(normal.x()) > abs(normal.z()),
        make_float3(-normal.y(), normal.x(), 0.0f),
        make_float3(0.0f, -normal.z(), normal.y())));
    Var tangent = normalize(cross(binormal, normal));
    Var<Onb> onb{tangent, binormal, normal};
    return onb;
}

inline Expr<float3> transform_to_local(Expr<Onb> onb, Expr<float3> v) noexcept {
    return make_float3(dot(v, onb.tangent()), dot(v, onb.binormal()), dot(v, onb.normal()));
}

inline Expr<float3> transform_to_world(Expr<Onb> onb, Expr<float3> v) noexcept {
    return v.x() * onb.tangent() + v.y() * onb.binormal() + v.z() * onb.normal();
}

inline Expr<float3> face_forward(Expr<float3> d_in, Expr<float3> ref_in) noexcept {
    Var d = d_in;
    Var ref = ref_in;
    return select(dot(d, ref) < 0.0f, -d, d);
}

inline Expr<float3> uniform_sample_hemisphere(Expr<float2> u) {
    Var r = sqrt(1.0f - u.x() * u.x());
    Var phi = 2.0f * constants::pi * u.y();
    return make_float3(r * cos(phi), r * sin(phi), u.x());
}

inline Expr<float3> cosine_sample_hemisphere(Expr<float2> u)
{
    Var r = sqrt(u.x());
    Var phi = 2.0f * constants::pi * u.y();
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x()));
}

}
}

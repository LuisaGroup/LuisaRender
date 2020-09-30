//
// Created by Mike Smith on 2020/9/16.
//

#pragma once

#include <compute/dsl_syntax.h>

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

inline Expr<float3> uniform_sample_hemisphere(Var<float2> u) {
    Var r = sqrt(1.0f - u.x * u.x);
    Var phi = 2.0f * constants::pi * u.y;
    return make_float3(r * cos(phi), r * sin(phi), u.x);
}

inline Expr<float3> cosine_sample_hemisphere(Var<float2> u) {
    Var r = sqrt(u.x);
    Var phi = 2.0f * constants::pi * u.y;
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
}

inline Expr<float2> uniform_sample_triangle(Var<float2> u) {
    Var sx = sqrt(u.x);
    return make_float2(1.0f - sx, u.y * sx);
}

template<typename Table, typename = decltype(std::declval<Table &>()[Expr{0u}])>
inline Expr<uint> sample_discrete(Table &&cdf, Var<uint> first, Var<uint> last, Var<float> u) noexcept {
    Var count = last - first;
    While (count > 0u) {
        Var step = count / 2;
        Var it = first + step;
        Var pred = cdf[it] < u;
        first = select(pred, it + 1u, first);
        count = select(pred, count - (step + 1u), step);
    };
    return min(first, last - 1u);
}

}

}

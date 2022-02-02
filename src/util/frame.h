//
// Created by Mike Smith on 2022/1/13.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using luisa::compute::Expr;
using luisa::compute::Bool;
using luisa::compute::Float;
using luisa::compute::Float3;

class Frame {

private:
    Float3 _u;
    Float3 _v;
    Float3 _n;

private:
    Frame(Float3 tangent, Float3 bitangent, Float3 normal) noexcept;

public:
    Frame() noexcept;
    [[nodiscard]] static Frame make(Expr<float3> normal) noexcept;
    [[nodiscard]] static Frame make(Expr<float3> normal, Expr<float3> tangent) noexcept;
    [[nodiscard]] Float3 local_to_world(Expr<float3> d) const noexcept;
    [[nodiscard]] Float3 world_to_local(Expr<float3> d) const noexcept;
    [[nodiscard]] Expr<float3> u() const noexcept { return _u; }
    [[nodiscard]] Expr<float3> v() const noexcept { return _v; }
    [[nodiscard]] Expr<float3> n() const noexcept { return _n; }
};

using compute::abs;
using compute::clamp;
using compute::cos;
using compute::def;
using compute::dot;
using compute::ite;
using compute::max;
using compute::sign;
using compute::sin;
using compute::sqrt;
using compute::saturate;

[[nodiscard]] inline auto sqr(auto x) noexcept { return x * x; }
[[nodiscard]] inline Float abs_dot(Float3 u, Float3 v) noexcept { return abs(dot(u, v)); }
[[nodiscard]] inline Float cos_theta(Float3 w) { return w.z; }
[[nodiscard]] inline Float cos2_theta(Float3 w) { return sqr(w.z); }
[[nodiscard]] inline Float abs_cos_theta(Float3 w) { return abs(w.z); }
[[nodiscard]] inline Float sin2_theta(Float3 w) { return saturate(1.0f - cos2_theta(w)); }
[[nodiscard]] inline Float sin_theta(Float3 w) { return sqrt(sin2_theta(w)); }
[[nodiscard]] inline Float tan_theta(Float3 w) { return sin_theta(w) / cos_theta(w); }
[[nodiscard]] inline Float tan2_theta(Float3 w) { return sin2_theta(w) / cos2_theta(w); }

[[nodiscard]] inline Float cos_phi(Float3 w) {
    auto sinTheta = sin_theta(w);
    return ite(sinTheta == 0.0f, 1.0f, clamp(w.x / sinTheta, -1.0f, 1.0f));
}

[[nodiscard]] inline Float sin_phi(Float3 w) {
    auto sinTheta = sin_theta(w);
    return ite(sinTheta == 0.0f, 0.0f, clamp(w.y / sinTheta, -1.0f, 1.0f));
}

[[nodiscard]] inline Float cos2_phi(Float3 w) { return sqr(cos_phi(w)); }
[[nodiscard]] inline Float sin2_phi(Float3 w) { return sqr(sin_phi(w)); }
[[nodiscard]] inline Bool same_hemisphere(Float3 w, Float3 wp) noexcept { return w.z * wp.z > 0.0f; }

}// namespace luisa::render

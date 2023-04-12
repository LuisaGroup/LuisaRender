//
// Created by Mike Smith on 2022/1/13.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using luisa::compute::Bool;
using luisa::compute::Expr;
using luisa::compute::Float;
using luisa::compute::Float3;

class Frame {

private:
    Float3 _s;
    Float3 _t;
    Float3 _n;

public:
    Frame() noexcept;
    Frame(Expr<float3> s, Expr<float3> t, Expr<float3> n) noexcept;
    void flip() noexcept;
    [[nodiscard]] static Frame make(Expr<float3> n) noexcept;
    [[nodiscard]] static Frame make(Expr<float3> n, Expr<float3> s) noexcept;
    [[nodiscard]] Float3 local_to_world(Expr<float3> d) const noexcept;
    [[nodiscard]] Float3 world_to_local(Expr<float3> d) const noexcept;
    [[nodiscard]] Expr<float3> s() const noexcept { return _s; }
    [[nodiscard]] Expr<float3> t() const noexcept { return _t; }
    [[nodiscard]] Expr<float3> n() const noexcept { return _n; }
};

using compute::abs;
using compute::clamp;
using compute::cos;
using compute::def;
using compute::dot;
using compute::ite;
using compute::max;
using compute::saturate;
using compute::sign;
using compute::sin;
using compute::sqrt;

[[nodiscard]] inline auto sqr(auto x) noexcept { return x * x; }
[[nodiscard]] inline auto one_minus_sqr(auto x) noexcept { return 1.f - sqr(x); }
[[nodiscard]] inline auto abs_dot(Expr<float3> u, Expr<float3> v) noexcept { return abs(dot(u, v)); }
[[nodiscard]] inline auto cos_theta(Expr<float3> w) { return w.z; }
[[nodiscard]] inline auto cos2_theta(Expr<float3> w) { return sqr(w.z); }
[[nodiscard]] inline auto abs_cos_theta(Expr<float3> w) { return abs(w.z); }
[[nodiscard]] inline auto sin2_theta(Expr<float3> w) { return saturate(1.0f - cos2_theta(w)); }
[[nodiscard]] inline auto sin_theta(Expr<float3> w) { return sqrt(sin2_theta(w)); }
[[nodiscard]] inline auto tan_theta(Expr<float3> w) { return sin_theta(w) / cos_theta(w); }
[[nodiscard]] inline auto tan2_theta(Expr<float3> w) { return sin2_theta(w) / cos2_theta(w); }

[[nodiscard]] inline auto cos_phi(Expr<float3> w) {
    auto sinTheta = sin_theta(w);
    return ite(sinTheta == 0.0f, 1.0f, clamp(w.x / sinTheta, -1.0f, 1.0f));
}

[[nodiscard]] inline auto sin_phi(Expr<float3> w) {
    auto sinTheta = sin_theta(w);
    return ite(sinTheta == 0.0f, 0.0f, clamp(w.y / sinTheta, -1.0f, 1.0f));
}

[[nodiscard]] inline auto cos2_phi(Expr<float3> w) { return sqr(cos_phi(w)); }
[[nodiscard]] inline auto sin2_phi(Expr<float3> w) { return sqr(sin_phi(w)); }
[[nodiscard]] inline auto same_hemisphere(Expr<float3> w, Expr<float3> wp) noexcept { return w.z * wp.z > 0.0f; }

// clamp the shading normal `ns` so that `w` and its reflection will go to the same hemisphere w.r.t. `ng`
[[nodiscard]] Float3 clamp_shading_normal(Expr<float3> ns, Expr<float3> ng, Expr<float3> w) noexcept;

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Frame)

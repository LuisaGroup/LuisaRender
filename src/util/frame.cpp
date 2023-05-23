//
// Created by Mike Smith on 2022/1/13.
//

#include <dsl/sugar.h>
#include <util/scattering.h>
#include <util/frame.h>

namespace luisa::render {

Frame::Frame(Expr<float3> s,
             Expr<float3> t,
             Expr<float3> n) noexcept
    : _s{s}, _t{t}, _n{n} {}

Frame::Frame() noexcept
    : _s{make_float3(1.f, 0.f, 0.f)},
      _t{make_float3(0.f, 1.f, 0.f)},
      _n{make_float3(0.f, 0.f, 1.f)} {}

Frame Frame::make(Expr<float3> n) noexcept {
    auto sgn = sign(n.z);
    auto a = -1.f / (sgn + n.z);
    auto b = n.x * n.y * a;
    auto s = make_float3(1.f + sgn * sqr(n.x) * a, sgn * b, -sgn * n.x);
    auto t = make_float3(b, sgn + sqr(n.y) * a, -n.y);
    return {normalize(s), normalize(t), n};
}

Frame Frame::make(Expr<float3> n, Expr<float3> s) noexcept {
    auto ss = normalize(s - n * dot(n, s));
    auto tt = normalize(cross(n, ss));
    return {ss, tt, n};
}

Float3 Frame::local_to_world(Expr<float3> d) const noexcept {
    return normalize(d.x * _s + d.y * _t + d.z * _n);
}

Float3 Frame::world_to_local(Expr<float3> d) const noexcept {
    return normalize(make_float3(dot(d, _s), dot(d, _t), dot(d, _n)));
}

void Frame::flip() noexcept {
    _n = -_n;
    _t = -_t;
}

Float3 clamp_shading_normal(Expr<float3> ns, Expr<float3> ng, Expr<float3> w) noexcept {
    auto w_refl = reflect(-w, ns);
    auto w_refl_clip = ite(dot(w_refl, ng) * dot(w, ng) > 0.f, w_refl,
                           normalize(w_refl - ng * dot(w_refl, ng)));
    return normalize(w_refl_clip + w);
}

}// namespace luisa::render

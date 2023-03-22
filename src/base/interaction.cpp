//
// Created by Mike Smith on 2022/12/3.
//

#include <base/interaction.h>

namespace luisa::render {

Bool Interaction::same_sided(Expr<float3> wo, Expr<float3> wi) const noexcept {
    return dot(wi, _ng) * dot(wo, _ng) > 0.0f;
}

Float3 Interaction::p_robust(Expr<float3> w) const noexcept {
    auto offset_factor = _shape.intersection_offset_factor();
    auto front = dot(_shading.n(), w) > 0.f;
    auto n = ite(front, _ng, -_ng);
//    return _pg + 1e-2f * normalize(n);
    return offset_ray_origin(_pg, offset_factor * n);
}

Var<Ray> Interaction::spawn_ray(Expr<float3> wi, Expr<float> t_max) const noexcept {
    return make_ray(p_robust(wi), wi, 0.f, t_max);
}

Var<Ray> Interaction::spawn_ray_to(Expr<float3> p) const noexcept {
    auto p_from = p_robust(p - _pg);
    auto L = p - p_from;
    auto d = length(L);
    return make_ray(p_from, L * (1.f / d), 0.f, d * .9999f);
}

}// namespace luisa::render

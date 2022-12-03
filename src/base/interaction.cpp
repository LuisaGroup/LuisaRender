//
// Created by Mike Smith on 2022/12/3.
//

#include <base/interaction.h>

namespace luisa::render {

Bool Interaction::same_sided(Expr<float3> wo, Expr<float3> wi) const noexcept {
    return dot(wi, _ng) * dot(wo, _ng) > 0.0f;
}

Float3 Interaction::p_robust(Expr<float3> w) const noexcept {
    auto offset_factor = clamp(_shape->intersection_offset_factor() * 255.f + 1.f, 1.f, 256.f);
    auto front = dot(_shading.n(), w) > 0.f;
    auto p = ite(front, _ps, _pg);
    auto n = ite(front, _ng, -_ng);
    return offset_ray_origin(p, offset_factor * n);
}

Var<Ray> Interaction::spawn_ray(Expr<float3> wi, Expr<float> t_max) const noexcept {
    return make_ray(p_robust(wi), wi, 0.f, t_max);
}

}// namespace luisa::render

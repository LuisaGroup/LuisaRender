//
// Created by Mike Smith on 2022/1/9.
//

#include <util/sampling.h>

namespace luisa::render {

using namespace luisa::compute;

Float2 sample_uniform_disk_concentric(Expr<float2> u_in) noexcept {
    auto u = u_in * 2.0f - 1.0f;
    auto p = abs(u.x) > abs(u.y);
    auto r = ite(p, u.x, u.y);
    auto theta = ite(p, pi_over_four * (u.y / u.x), pi_over_two - pi_over_four * (u.x / u.y));
    return r * make_float2(cos(theta), sin(theta));
}

Float3 sample_cosine_hemisphere(Expr<float2> u) noexcept {
    auto d = sample_uniform_disk_concentric(u);
    auto z = sqrt(max(1.0f - d.x * d.x - d.y * d.y, 0.0f));
    return make_float3(d.x, d.y, z);
}

Float cosine_hemisphere_pdf(Expr<float> cos_theta) noexcept {
    return cos_theta * inv_pi;
}

}// namespace luisa::render

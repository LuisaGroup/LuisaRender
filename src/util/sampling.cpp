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

std::pair<luisa::vector<AliasEntry>, luisa::vector<float>>
create_alias_table(luisa::span<float> values) noexcept {

    auto sum = std::reduce(values.cbegin(), values.cend(), 0.0);
    auto inv_sum = 1.0 / sum;
    luisa::vector<float> pdf(values.size());
    std::transform(
        values.cbegin(), values.cend(), pdf.begin(),
        [inv_sum](auto v) noexcept {
            return static_cast<float>(v * inv_sum);
        });

    auto ratio = static_cast<double>(values.size()) / sum;
    static thread_local luisa::vector<uint> over;
    static thread_local luisa::vector<uint> under;
    over.clear();
    under.clear();
    over.reserve(next_pow2(values.size()));
    under.reserve(next_pow2(values.size()));

    luisa::vector<AliasEntry> table(values.size());
    for (auto i = 0u; i < values.size(); i++) {
        auto p = static_cast<float>(values[i] * ratio);
        table[i] = {p, i};
        (p > 1.0f ? over : under).emplace_back(i);
    }

    while (!over.empty() && !under.empty()) {
        auto o = over.back();
        auto u = under.back();
        over.pop_back();
        under.pop_back();
        table[o].prob -= 1.0f - table[u].prob;
        table[u].alias = o;
        if (table[o].prob > 1.0f) {
            over.push_back(o);
        } else if (table[o].prob < 1.0f) {
            under.push_back(o);
        }
    }
    for (auto i : over) { table[i] = {1.0f, i}; }
    for (auto i : under) { table[i] = {1.0f, i}; }

    return std::make_pair(std::move(table), std::move(pdf));
}

Float3 sample_uniform_triangle(Expr<float2> u) noexcept {
    auto uv = ite(
        u.x < u.y,
        make_float2(0.5f * u.x, -0.5f * u.x + u.y),
        make_float2(-0.5f * u.y + u.x, 0.5f * u.y));
    return make_float3(uv, 1.0f - uv.x - uv.y);
}

Float3 sample_uniform_sphere(Expr<float2> u) noexcept {
    auto z = 1.0f - 2.0f * u.x;
    auto r = sqrt(max(1.0f - z * z, 0.0f));
    auto phi = 2.0f * pi * u.y;
    return make_float3(r * cos(phi), r * sin(phi), z);
}

Float2 invert_uniform_sphere_sample(Expr<float3> w) noexcept {
    auto phi = atan2(w.y, w.x);
    phi = ite(phi < 0.0f, phi + pi * 2.0f, phi);
    return make_float2(0.5f * (1.0f - w.z), phi * (0.5f * inv_pi));
}

}// namespace luisa::render

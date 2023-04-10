//
// Created by Mike Smith on 2022/1/9.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <dsl/sugar.h>

namespace luisa::render {

using namespace luisa::compute;

Float2 sample_uniform_disk_concentric(Expr<float2> u) noexcept {
    static Callable impl = [](Float2 u_in) noexcept {
        auto u = u_in * 2.0f - 1.0f;
        auto p = abs(u.x) > abs(u.y);
        auto r = ite(p, u.x, u.y);
        auto theta = ite(p, pi_over_four * (u.y / u.x), pi_over_two - pi_over_four * (u.x / u.y));
        return r * make_float2(cos(theta), sin(theta));
    };
    return impl(u);
}

Float3 sample_cosine_hemisphere(Expr<float2> u) noexcept {
    static Callable impl = [](Float2 u) noexcept {
        auto d = sample_uniform_disk_concentric(u);
        auto z = sqrt(max(1.0f - d.x * d.x - d.y * d.y, 0.0f));
        return make_float3(d.x, d.y, z);
    };
    return impl(u);
}

Float cosine_hemisphere_pdf(Expr<float> cos_theta) noexcept {
    return cos_theta * inv_pi;
}

std::pair<luisa::vector<AliasEntry>, luisa::vector<float>>
create_alias_table(luisa::span<const float> values) noexcept {

    auto sum = 0.0;
    for (auto v : values) { sum += std::abs(v); }
    luisa::vector<float> pdf(values.size());
    if (sum == 0.) [[unlikely]] {
        auto n = static_cast<double>(values.size());
        std::fill(pdf.begin(), pdf.end(), static_cast<float>(1.0 / n));
    } else [[likely]] {
        auto inv_sum = 1.0 / sum;
        std::transform(
            values.cbegin(), values.cend(), pdf.begin(),
            [inv_sum](auto v) noexcept {
                return static_cast<float>(std::abs(v) * inv_sum);
            });
    }

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
    static Callable impl = [](Float2 u) noexcept {
        auto uv = ite(
            u.x < u.y,
            make_float2(0.5f * u.x, -0.5f * u.x + u.y),
            make_float2(-0.5f * u.y + u.x, 0.5f * u.y));
        return make_float3(uv, 1.0f - uv.x - uv.y);
    };
    return impl(u);
}

Float3 sample_uniform_sphere(Expr<float2> u) noexcept {
    static Callable impl = [](Float2 u) noexcept {
        auto z = 1.0f - 2.0f * u.x;
        auto r = sqrt(max(1.0f - z * z, 0.0f));
        auto phi = 2.0f * pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), z);
    };
    return impl(u);
}

Float2 invert_uniform_sphere_sample(Expr<float3> w) noexcept {
    static Callable impl = [](Float3 w) noexcept {
        auto phi = atan2(w.y, w.x);
        phi = ite(phi < 0.0f, phi + pi * 2.0f, phi);
        return make_float2(0.5f * (1.0f - w.z), phi * (0.5f * inv_pi));
    };
    return impl(w);
}

Float uniform_cone_pdf(Expr<float> cos_theta_max) noexcept {
    return 1.f / (2.f * pi * (1.f - cos_theta_max));
}

Float3 sample_uniform_cone(Expr<float2> u, Expr<float> cos_theta_max) noexcept {
    static Callable impl = [](Float2 u, Float cos_theta_max) noexcept {
        auto cosTheta = (1 - u.x) + u.x * cos_theta_max;
        auto sinTheta = sqrt(max(1.f - cosTheta * cosTheta, 0.f));
        auto phi = 2.f * pi * u.y;
        return spherical_direction(sinTheta, cosTheta, phi);
    };
    return impl(u, cos_theta_max);
}

Float balance_heuristic(Expr<uint> nf, Expr<float> fPdf, Expr<uint> ng, Expr<float> gPdf) noexcept {
    static Callable impl = [](UInt nf, Float fPdf, UInt ng, Float gPdf) noexcept {
        auto sum_f = nf * fPdf;
        auto sum = sum_f + ng * gPdf;
        return ite(sum == 0.0f, 0.0f, sum_f / sum);
    };
    return impl(nf, fPdf, ng, gPdf);
}

Float power_heuristic(Expr<uint> nf, Expr<float> fPdf, Expr<uint> ng, Expr<float> gPdf) noexcept {
    static Callable impl = [](UInt nf, Float fPdf, UInt ng, Float gPdf) noexcept {
        Float f = nf * fPdf, g = ng * gPdf;
        auto ff = f * f;
        auto gg = g * g;
        auto sum = ff + gg;
        return ite(isinf(ff), 1.f, ite(sum == 0.f, 0.f, ff / sum));
    };
    return impl(nf, fPdf, ng, gPdf);
}

Float balance_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept {
    return balance_heuristic(1u, fPdf, 1u, gPdf);
}

Float power_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept {
    return power_heuristic(1u, fPdf, 1u, gPdf);
}


UInt sample_discrete(Expr<float2> weights, Expr<float> u) noexcept {
    Float u_rescaled = u * (weights.x + weights.y);
    auto ans = ite(u_rescaled <= weights.x, 0u, 1u);
    return ans;
}

UInt sample_discrete(Expr<float3> weights, Expr<float> u) noexcept {
    UInt ans = def<uint>(-1);
    Float accum_sum = 0.0f;
    Float u_rescaled = u * (weights.x + weights.y + weights.z);
    $for(i, 3u) {
        accum_sum += weights[i];
        $if(u_rescaled <= accum_sum) {
            ans = i;
            $break;
        };
    };
    return ans;
}

UInt sample_discrete(const SampledSpectrum &weights, Expr<float> u) noexcept {
    UInt ans = def<uint>(-1);
    Float accum_sum = 0.0f;
    Float u_rescaled = u * weights.sum();
    $for(i, weights.dimension()) {
        accum_sum += weights[i];
        $if(u_rescaled <= accum_sum) {
            ans = i;
            $break;
        };
    };
    return ans;
}

Float sample_exponential(Expr<float> u, Expr<float> a) noexcept {
    return -log(1.f - u) / a;
}

}// namespace luisa::render

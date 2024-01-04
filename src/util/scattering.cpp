//
// Created by Mike Smith on 2022/1/31.
//

#include "dsl/builtin.h"
#include "dsl/stmt.h"
#include "util/spec.h"
#include <dsl/sugar.h>
#include <util/frame.h>
#include <util/sampling.h>
#include <util/scattering.h>

namespace luisa::render {

using compute::backward;
using compute::Callable;
using compute::grad;
using compute::requires_grad;

Bool refract(Float3 wi, Float3 n, Float eta, Float3 *wt) noexcept {
    static Callable impl = [](Float3 wi, Float3 n, Float eta) noexcept {
        // Compute $\cos \theta_\roman{t}$ using Snell's law
        auto cosThetaI = dot(n, wi);
        auto sin2ThetaI = max(0.0f, one_minus_sqr(cosThetaI));
        auto sin2ThetaT = sqr(eta) * sin2ThetaI;
        auto cosThetaT = sqrt(1.f - sin2ThetaT);
        // Handle total internal reflection for transmission
        auto wt = (eta * cosThetaI - cosThetaT) * n - eta * wi;
        return make_float4(wt, sin2ThetaT);
    };
    auto v = impl(wi, n, eta);
    *wt = v.xyz();
    return v.w < 1.0f;
}

Float fresnel_dielectric(Float cosThetaI_in, Float etaI_in, Float etaT_in) noexcept {
    static Callable impl = [](Float cosThetaI_in, Float etaI_in, Float etaT_in) noexcept {
        using namespace compute;
        auto cosThetaI = clamp(cosThetaI_in, -1.f, 1.f);
        // Potentially swap indices of refraction
        auto entering = cosThetaI > 0.f;
        auto etaI = ite(entering, etaI_in, etaT_in);
        auto etaT = ite(entering, etaT_in, etaI_in);
        cosThetaI = abs(cosThetaI);
        // Compute _cosThetaT_ using Snell's law
        auto sinThetaI = sqrt(max(0.f, one_minus_sqr(cosThetaI)));
        auto sinThetaT = etaI / etaT * sinThetaI;
        auto cosThetaT = sqrt(max(0.f, one_minus_sqr(sinThetaT)));
        auto Rparl = (etaT * cosThetaI - etaI * cosThetaT) /
                     (etaT * cosThetaI + etaI * cosThetaT);
        auto Rperp = (etaI * cosThetaI - etaT * cosThetaT) /
                     (etaI * cosThetaI + etaT * cosThetaT);
        auto fr = (Rparl * Rparl + Rperp * Rperp) * .5f;
        // Handle total internal reflection
        return ite(sinThetaT < 1.f, fr, 1.f);
    };
    return impl(cosThetaI_in, etaI_in, etaT_in);
}

SampledSpectrum fresnel_conductor(
    Float cosThetaI, Float etai, const SampledSpectrum &etat, const SampledSpectrum &k) noexcept {
    using namespace compute;
    cosThetaI = clamp(cosThetaI, -1.f, 1.f);
    auto eta = etat / etai;
    auto etak = k / etai;
    auto cosThetaI2 = cosThetaI * cosThetaI;
    auto sinThetaI2 = 1.f - cosThetaI2;
    auto eta2 = eta * eta;
    auto etak2 = etak * etak;
    auto t0 = eta2 - etak2 - sinThetaI2;
    auto a2plusb2 = sqrt(t0 * t0 + 4.f * eta2 * etak2);
    auto t1 = a2plusb2 + cosThetaI2;
    auto a = sqrt(.5f * (a2plusb2 + t0));
    auto t2 = 2.f * cosThetaI * a;
    auto Rs = (t1 - t2) / (t1 + t2);
    auto t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    auto t4 = t2 * sinThetaI2;
    auto Rp = Rs * (t3 - t4) / (t3 + t4);
    return .5f * (Rp + Rs);
}

Float3 face_forward(Float3 v, Float3 n) noexcept {
    return ite(dot(v, n) < 0.f, -v, v);
}

Float3 spherical_direction(Float sinTheta, Float cosTheta, Float phi) noexcept {
    return make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

Float3 spherical_direction(Float sinTheta, Float cosTheta, Float phi, Float3 x, Float3 y, Float3 z) noexcept {
    return sinTheta * cos(phi) * x + sinTheta * sin(phi) * y + cosTheta * z;
}

Float spherical_theta(Float3 v) noexcept {
    return acos(clamp(v.z, -1.f, 1.f));
}

Float spherical_phi(Float3 v) noexcept {
    auto p = atan2(v.y, v.x);
    return ite(p < 0.f, p + 2.f * pi, p);
}

Float fresnel_dielectric_integral(Float eta) noexcept {
    static Callable fit_less_one = [](Float eta) noexcept {
        constexpr std::array c{0.75985009f, -2.09069066f, 2.23559031f, -0.90663979f};
        return polynomial(eta, c[0], c[1], c[2], c[3]);
    };
    static Callable fit_greater_one = [](Float eta) noexcept {
        constexpr std::array c{0.97945724f, 0.21762732f, -1.18995376f};
        return polynomial(1.f / eta, c[0], c[1], c[2]);
    };
    return saturate(ite(eta == 1.f, 0.f, ite(eta < 1.f, fit_less_one(eta), fit_greater_one(eta))));
}

Float MicrofacetDistribution::G1(Expr<float3> w) const noexcept {
    return 1.0f / (1.0f + Lambda(w));
}

Float MicrofacetDistribution::G(Expr<float3> wo, Expr<float3> wi) const noexcept {
    return forward_compute_G(wo, wi, alpha());
}

Float MicrofacetDistribution::forward_compute_G(Expr<float3> wo, Expr<float3> wi, Float2 alpha) const noexcept {
    return 1.0f / (1.0f + forward_compute_Lambda(wo, alpha) + forward_compute_Lambda(wi, alpha));
}

Float MicrofacetDistribution::pdf(Expr<float3> wo, Expr<float3> wh) const noexcept {
    using compute::abs;
    using compute::dot;
    return D(wh) * G1(wo) * abs_dot(wo, wh) / abs_cos_theta(wo);
}

MicrofacetDistribution::MicrofacetDistribution(Expr<float2> alpha) noexcept
    : _alpha{compute::max(alpha, 1e-4f)} {}

MicrofacetDistribution::Gradient MicrofacetDistribution::grad_G1(Expr<float3> w) const noexcept {
    auto d_alpha = -grad_Lambda(w).dAlpha / sqr(1.0f + Lambda(w));
    return {.dAlpha = d_alpha};
}

MicrofacetDistribution::Gradient MicrofacetDistribution::grad_G(Expr<float3> wo, Expr<float3> wi) const noexcept {
    auto d_alpha = -(grad_Lambda(wo).dAlpha + grad_Lambda(wi).dAlpha) / sqr(1.0f + Lambda(wo) + Lambda(wi));
    return {.dAlpha = d_alpha};
}

TrowbridgeReitzDistribution::TrowbridgeReitzDistribution(Expr<float2> alpha) noexcept
    : MicrofacetDistribution{alpha} {}

Float TrowbridgeReitzDistribution::roughness_to_alpha(Expr<float> roughness) noexcept {
    return max(sqr(roughness), 1e-4f);
}

Float2 TrowbridgeReitzDistribution::roughness_to_alpha(Expr<float2> roughness) noexcept {
    return max(sqr(roughness), 1e-4f);
}

Float TrowbridgeReitzDistribution::alpha_to_roughness(Expr<float> alpha) noexcept {
    return sqrt(max(alpha, 1e-4f));
}

Float2 TrowbridgeReitzDistribution::alpha_to_roughness(Expr<float2> alpha) noexcept {
    return sqrt(max(alpha, 1e-4f));
}

Float2 TrowbridgeReitzDistribution::grad_alpha_roughness(Expr<float2> roughness) noexcept {
    return 2.f * roughness;
}

Float TrowbridgeReitzDistribution::D(Expr<float3> wh) const noexcept {
    return forward_compute_D(wh, alpha());
}

Float TrowbridgeReitzDistribution::forward_compute_D(Expr<float3> wh, Float2 alpha) const noexcept {
    using compute::isinf;
    static Callable impl = [](Float3 wh, Float2 alpha) noexcept {
        auto tan2Theta = tan2_theta(wh);
        auto cos4Theta = sqr(cos2_theta(wh));
        auto e = tan2Theta * (sqr(cos_phi(wh) / alpha.x) +
                              sqr(sin_phi(wh) / alpha.y));
        auto d = 1.0f / (pi * alpha.x * alpha.y * cos4Theta * sqr(1.f + e));
        return ite(isinf(tan2Theta), 0.f, d);
    };
    return impl(wh, alpha);
}

Float TrowbridgeReitzDistribution::Lambda(Expr<float3> w) const noexcept {
    return forward_compute_Lambda(w, alpha());
}

Float TrowbridgeReitzDistribution::forward_compute_Lambda(Expr<float3> w, Float2 alpha) const noexcept {
    using compute::isinf;
    static Callable impl = [](Float3 w, Float2 alpha) noexcept {
        auto tanTheta = abs(tan_theta(w));
        // Compute _alpha_ for direction _w_
        auto alpha2 = cos2_phi(w) * sqr(alpha.x) +
                      sin2_phi(w) * sqr(alpha.y);
        auto alpha2Tan2Theta = alpha2 * sqr(tanTheta);
        auto L = (-1.f + sqrt(1.f + alpha2Tan2Theta)) * .5f;
        return ite(isinf(tanTheta), 0.f, L);
    };
    return impl(w, alpha);
}

[[nodiscard]] inline Float2 TrowbridgeReitzSample11(Expr<float> cosTheta, Expr<float2> U) noexcept {
    using namespace luisa::compute;
    static Callable impl = [](Float cosTheta, Float2 U) noexcept {
        auto slope = def(make_float2());
        $if(cosTheta <= .9999f) {
            auto sinTheta = sqrt(max(0.f, 1.f - sqr(cosTheta)));
            auto tanTheta = sinTheta / cosTheta;
            auto a = 1.f / tanTheta;
            auto G1 = 2.f / (1.f + sqrt(1.f + 1.f / sqr(a)));

            // sample slope_x
            auto A = 2.f * U.x / G1 - 1.f;
            auto tmp = min(1.f / (sqr(A) - 1.f), 1e10f);
            auto B = tanTheta;
            auto D = sqrt(max(sqr(B * tmp) - (sqr(A) - sqr(B)) * tmp, 0.f));
            auto slope_x_1 = B * tmp - D;
            auto slope_x_2 = B * tmp + D;
            auto slope_x = ite(
                (A < 0.f) | (slope_x_2 * tanTheta > 1.f),
                slope_x_1, slope_x_2);

            // sample slope_y
            auto S = ite(U.y > .5f, 1.f, -1.f);
            auto U2 = ite(U.y > .5f, 2.f * (U.y - .5f), 2.f * (.5f - U.y));
            auto z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
                     (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
            auto slope_y = S * z * sqrt(1.f + sqr(slope_x));
            slope = make_float2(slope_x, slope_y);
        }
        $else {// special case (normal incidence)
            auto r = sqrt(U.x / (1.f - U.x));
            auto phi = (2.f * pi) * U.y;
            slope = r * make_float2(cos(phi), sin(phi));
        };
        return slope;
    };
    return impl(cosTheta, U);
}

[[nodiscard]] inline Float3 TrowbridgeReitzSample(Expr<float3> wi, Expr<float2> alpha, Expr<float2> U) noexcept {
    using compute::make_float2;
    using compute::make_float3;
    using compute::normalize;
    static Callable impl = [](Float3 wi, Float2 alpha, Float2 U) noexcept {
        // 1. stretch wi
        auto wiStretched = normalize(make_float3(
            alpha.x * wi.x, alpha.y * wi.y, wi.z));
        // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
        auto slope = TrowbridgeReitzSample11(cos_theta(wiStretched), U);
        // 3. rotate
        slope = make_float2(
            cos_phi(wiStretched) * slope.x - sin_phi(wiStretched) * slope.y,
            sin_phi(wiStretched) * slope.x + cos_phi(wiStretched) * slope.y);
        // 4. unstretch
        slope = alpha * slope;
        // 5. compute normal
        return normalize(make_float3(-slope, 1.f));
    };
    return impl(wi, alpha, U);
}

Float3 TrowbridgeReitzDistribution::sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept {
    auto s = sign(cos_theta(wo));
    auto wh = TrowbridgeReitzSample(s * wo, alpha(), u);
    return s * wh;
}

MicrofacetDistribution::Gradient TrowbridgeReitzDistribution::grad_Lambda(Expr<float3> w) const noexcept {
    using compute::isinf;
    auto absTanTheta = abs(tan_theta(w));

    auto alpha2 = cos2_phi(w) * sqr(alpha().x) +
                  sin2_phi(w) * sqr(alpha().y);
    auto alpha2Tan2Theta = alpha2 * sqr(absTanTheta);

    auto d_Lambda = ite(isinf(absTanTheta), 0.f, 1.f);
    auto d_alpha2Tan2Theta = d_Lambda * .25f / sqrt(1.f + alpha2Tan2Theta);
    auto d_alpha2 = d_alpha2Tan2Theta * sqr(absTanTheta);
    auto d_alpha = d_alpha2 * make_float2(.5f * cos2_phi(w) / sqr(alpha().x),
                                          .5f * sin2_phi(w) / sqr(alpha().y));

    return {.dAlpha = d_alpha};
}

MicrofacetDistribution::Gradient TrowbridgeReitzDistribution::grad_D(Expr<float3> wh) const noexcept {
    auto d_alpha = alpha();
    $autodiff {
        auto alpha_back = alpha();
        requires_grad(alpha_back);
        auto y = forward_compute_D(wh, alpha_back);
        backward(y);
        d_alpha = grad(alpha_back);
    };
    return {.dAlpha = d_alpha};
}

SampledSpectrum FresnelConductor::evaluate(Expr<float> cosThetaI) const noexcept {
    return fresnel_conductor(abs(cosThetaI), _eta_i, _eta_t, _k);
}

SampledSpectrum FresnelDielectric::evaluate(Expr<float> cosThetaI) const noexcept {
    return SampledSpectrum{fresnel_dielectric(cosThetaI, _eta_i, _eta_t)};
}

SampledSpectrum BxDF::sample(Expr<float3> wo, Float3 *wi, Expr<float2> u,
                             Float *p, TransportMode mode) const noexcept {
    auto wi_sample = sample_wi(wo, u, mode);
    auto valid = wi_sample.valid;
    *wi = wi_sample.wi;
    *p = ite(valid, pdf(wo, *wi, mode), 0.f);
    return ite(valid, evaluate(wo, *wi, mode), SampledSpectrum{0.f});
}

Float BxDF::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    return compute::ite(same_hemisphere(wo, wi), abs_cos_theta(wi) * inv_pi, 0.f);
}

BxDF::SampledDirection BxDF::sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept {
    auto wi = sample_cosine_hemisphere(u);
    wi.z *= compute::sign(cos_theta(wo));
    return {.wi = wi, .valid = true};
}

SampledSpectrum LambertianReflection::evaluate(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    return _r * ite(same_hemisphere(wo, wi), inv_pi, 0.f);
}

LambertianReflection::Gradient LambertianReflection::backward(
    Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode) const noexcept {
    return {.dR = df * ite(same_hemisphere(wo, wi), inv_pi, 0.f)};
}

SampledSpectrum LambertianTransmission::evaluate(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    return _t * ite(!same_hemisphere(wo, wi), inv_pi, 0.f);
}

BxDF::SampledDirection LambertianTransmission::sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept {
    auto wi = sample_cosine_hemisphere(u);
    wi.z *= -compute::sign(cos_theta(wo));
    return {.wi = wi, .valid = true};
}

Float LambertianTransmission::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    return compute::ite(same_hemisphere(wo, wi), 0.0f, abs_cos_theta(wi) * inv_pi);
}

LambertianTransmission::Gradient LambertianTransmission::backward(
    Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode) noexcept {
    return {.dT = df * ite(!same_hemisphere(wo, wi), inv_pi, 0.f)};
}

SampledSpectrum MicrofacetReflection::evaluate(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    return forward_compute(wo, wi, mode, _r, _distribution->alpha());
}

SampledSpectrum MicrofacetReflection::forward_compute(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode, SampledSpectrum r, Float2 alpha) const noexcept {
    using compute::any;
    using compute::normalize;
    auto wh = wi + wo;
    SampledSpectrum f{_r.dimension()};
    // TODO: autodiff do not support $if ?
    $if(same_hemisphere(wo, wi) & any(wh != 0.f)) {
        //     wh = normalize(wh);
        // For the Fresnel call, make sure that wh is in the same hemisphere
        // as the surface normal, so that TIR is handled correctly.
        auto F = _fresnel->evaluate(dot(wi, face_forward(wh, make_float3(0.f, 0.f, 1.f))));
        auto D = _distribution->forward_compute_D(wh, alpha);
        auto G = _distribution->forward_compute_G(wo, wi, alpha);
        auto cos_o = cos_theta(wo);
        auto cos_i = cos_theta(wi);
        f = r * F * abs(0.25f * D * G / (cos_i * cos_o));
    };
    return f;
}

BxDF::SampledDirection MicrofacetReflection::sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept {
    auto wh = _distribution->sample_wh(wo, u);
    auto wi = reflect(-wo, wh);
    return {.wi = wi, .valid = same_hemisphere(wo, wi)};
}

Float MicrofacetReflection::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    auto p = def(0.f);
    auto wh = wi + wo;
    $if(same_hemisphere(wo, wi) & any(wh != 0.f)) {
        wh = normalize(wh);
        p = _distribution->pdf(wo, wh) / (4.f * dot(wo, wh));
    };
    return p;
}

MicrofacetReflection::Gradient MicrofacetReflection::backward(
    Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode) const noexcept {
    auto d_alpha = _distribution->alpha();
    auto d_r = _r;
    $autodiff {
        auto alpha_back = _distribution->alpha();
        auto r = _r;
        requires_grad(alpha_back);
        r.requires_grad();
        auto y = forward_compute(wo, wi, mode, r, alpha_back);
        y.backward(df);
        d_alpha = grad(alpha_back);
        d_r = r.grad();
    };

    using compute::any;
    using compute::normalize;
    auto wh = wi + wo;
    auto valid = same_hemisphere(wo, wi) & any(wh != 0.f);
    wh = normalize(wh);
    auto cosI_eval = dot(wi, face_forward(wh, make_float3(0.f, 0.f, 1.f)));
    auto F = _fresnel->evaluate(cosI_eval);
    auto D = _distribution->D(wh);
    auto G = _distribution->G(wo, wi);
    auto cos_o = cos_theta(wo);
    auto cos_i = cos_theta(wi);
    auto k0 = 0.25f / (cos_i * cos_o);
    auto k1 = k0 * D * G;
    auto h = abs(k1);

    // // backward
    // auto d_h = (df * _r * F).sum() * ite(valid, 1.f, 0.f);
    auto d_F = df * _r * ite(valid, h, 0.f);
    // auto k2 = d_h * sign(k1) * k0;
    // auto d_D = k2 * G;
    // auto d_G = k2 * D;
    // auto d_alpha = d_D * _distribution->grad_D(wh).dAlpha +
    //                d_G * _distribution->grad_G(wo, wi).dAlpha;
    // auto d_r = df * F * ite(valid, h, 0.f);

    return {.dR = d_r, .dAlpha = d_alpha, .dFresnel = _fresnel->backward(cosI_eval, d_F)};
}

SampledSpectrum MicrofacetTransmission::evaluate(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    auto cosThetaO = cos_theta(wo);
    auto cosThetaI = cos_theta(wi);
    auto eta = ite(cosThetaO > 0.f, _eta_b / _eta_a, _eta_a / _eta_b);
    auto wh = normalize(wo + wi * eta);
    wh = compute::sign(cos_theta(wh)) * wh;
    SampledSpectrum f{_t.dimension()};
    $if(!same_hemisphere(wo, wi) &
        cosThetaO != 0.f & cosThetaI != 0.f &
        dot(wo, wh) * dot(wi, wh) < 0.f) {
        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        auto G = _distribution->G(wo, wi);
        auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        auto F = fresnel_dielectric(dot(wo, wh), _eta_a, _eta_b);
        auto D = _distribution->D(wh);
        f = (1.f - F) * _t * D * G * dot(wi, wh) * dot(wo, wh) /
            (cosThetaI * cosThetaO * sqr(sqrtDenom));
        if (mode == TransportMode::IMPORTANCE) {
            f *= sqr(eta);
        }
    };
    return f;
}

BxDF::SampledDirection MicrofacetTransmission::sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept {
    auto eta = ite(cos_theta(wo) > 0.f, _eta_a / _eta_b, _eta_b / _eta_a);
    auto wh = _distribution->sample_wh(wo, u);
    auto wi = compute::def(make_float3(0.f));
    auto refr = refract(wo, wh, eta, std::addressof(wi));
    return {.wi = wi, .valid = refr & !same_hemisphere(wo, wi)};
}

Float MicrofacetTransmission::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    auto pdf = def(0.f);
    auto entering = cos_theta(wo) > 0.f;
    auto eta = ite(entering, _eta_b / _eta_a, _eta_a / _eta_b);
    auto wh = normalize(wo + wi * eta);
    $if(!same_hemisphere(wo, wi) & dot(wo, wh) * dot(wi, wh) < 0.f) {
        // Compute change of variables _dwh\_dwi_ for microfacet transmission
        auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        auto dwh_dwi = sqr(eta / sqrtDenom) * abs_dot(wi, wh);
        auto valid = dot(wo, wh) * dot(wi, wh) < 0.f;
        pdf = ite(valid, _distribution->pdf(wo, wh) * dwh_dwi, 0.f);
    };
    return pdf;
}

MicrofacetTransmission::Gradient MicrofacetTransmission::backward(
    Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode) const noexcept {
    auto cosThetaO = cos_theta(wo);
    auto cosThetaI = cos_theta(wi);
    auto refr = !same_hemisphere(wo, wi) & cosThetaO != 0.f & cosThetaI != 0.f;
    auto e = ite(cosThetaO > 0.f, _eta_b / _eta_a, _eta_a / _eta_b);

    auto G = _distribution->G(wo, wi);
    auto grad_G = _distribution->grad_G(wo, wi);
    auto d_t = SampledSpectrum{df.dimension(), 0.f};
    auto wh = normalize(wo + wi * e);
    wh = compute::sign(cos_theta(wh)) * wh;
    auto valid = refr & dot(wo, wh) * dot(wi, wh) < 0.f;
    auto sqrtDenom = dot(wo, wh) + e * dot(wi, wh);
    auto factor = 1.f / e;

    auto F = fresnel_dielectric(dot(wo, wh), _eta_a, _eta_b);
    auto D = _distribution->D(wh);
    auto k0 = dot(wi, wh) * dot(wo, wh) /
              (cosThetaI * cosThetaO * sqr(sqrtDenom));
    if (mode == TransportMode::IMPORTANCE) { k0 *= sqr(e); }
    auto k1 = D * G * k0;
    auto k2 = (1.f - F) * sqr(factor);
    auto f = k2 * _t * abs(k1);

    auto d_f = ite(valid, df, 0.f);
    d_t = d_f * k2 * abs(k1);
    auto grad_D = _distribution->grad_D(wh);
    auto d_alpha = def(make_float2(0.f));
    for (auto i = 0u; i < _t.dimension(); i++) {
        d_alpha += d_f[i] * k2 * _t[i] * sign(k1) * k0 *
                   (D * grad_G.dAlpha + G * grad_D.dAlpha);
    }
    return {.dT = d_t, .dAlpha = d_alpha};
}

OrenNayar::OrenNayar(const SampledSpectrum &R, Expr<float> sigma) noexcept
    : _r{R}, _sigma{sigma} {
    auto sigma2 = sqr(radians(sigma));
    _a = 1.f - (sigma2 / (2.f * sigma2 + 0.66f));
    _b = 0.45f * sigma2 / (sigma2 + 0.09f);
}

SampledSpectrum OrenNayar::evaluate(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    return forward_compute(wo, wi, mode, _a, _b, _r);
}

OrenNayar::Gradient OrenNayar::eval_grad(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    auto d_r = _r;
    auto d_sigma = _sigma;
    $autodiff {
        auto r = _r;
        auto sigma = _sigma;
        r.requires_grad();
        requires_grad(sigma);
        auto sigma2 = sqr(radians(sigma));
        auto a = 1.f - (sigma2 / (2.f * sigma2 + 0.66f));
        auto b = 0.45f * sigma2 / (sigma2 + 0.09f);
        auto y = forward_compute(wo, wi, mode, a, b, r);
        y.backward();
        d_r = r.grad();
        d_sigma = grad(sigma);
    };
    return {.dR = d_r, .dSigma = d_sigma};
}

SampledSpectrum OrenNayar::forward_compute(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode, Float a, Float b, SampledSpectrum r) const noexcept {
    auto valid = same_hemisphere(wo, wi);
    auto s = ite(valid, inv_pi, 0.f);
    static Callable scale = [](Float3 wo, Float3 wi, Float a, Float b) noexcept {
        auto sinThetaI = sin_theta(wi);
        auto sinThetaO = sin_theta(wo);
        // Compute cosine term of Oren-Nayar model
        auto sinPhiI = sin_phi(wi);
        auto cosPhiI = cos_phi(wi);
        auto sinPhiO = sin_phi(wo);
        auto cosPhiO = cos_phi(wo);
        auto dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
        auto maxCos = ite(sinThetaI > 1e-4f & sinThetaO > 1e-4f, max(0.f, dCos), 0.f);
        // Compute sine and tangent terms of Oren-Nayar model
        auto absCosThetaI = abs_cos_theta(wi);
        auto absCosThetaO = abs_cos_theta(wo);
        auto sinAlpha = ite(absCosThetaI > absCosThetaO, sinThetaO, sinThetaI);
        auto tanBeta = ite(absCosThetaI > absCosThetaO,
                           sinThetaI / absCosThetaI, sinThetaO / absCosThetaO);
        return (a + b * maxCos * sinAlpha * tanBeta);
    };
    return s * scale(wo, wi, a, b) * r;
}

OrenNayar::Gradient OrenNayar::backward(
    Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode) const noexcept {
    auto d_r = _r;
    auto d_sigma = _sigma;
    $autodiff {
        auto r = _r;
        auto sigma = _sigma;
        r.requires_grad();
        requires_grad(sigma);
        auto sigma2 = sqr(radians(sigma));
        auto a = 1.f - (sigma2 / (2.f * sigma2 + 0.66f));
        auto b = 0.45f * sigma2 / (sigma2 + 0.09f);
        auto y = forward_compute(wo, wi, mode, a, b, r);
        y.backward(df);
        d_r = r.grad();
        d_sigma = grad(sigma);
    };
    return {.dR = d_r, .dSigma = d_sigma};
}

SampledSpectrum FresnelBlend::Schlick(Expr<float> cosTheta) const noexcept {
    auto pow5 = [](auto &&v) { return sqr(sqr(v)) * v; };
    return _rs + pow5(1.f - cosTheta) * (1.f - _rs);
}

SampledSpectrum FresnelBlend::evaluate(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    auto wh = wi + wo;
    auto valid = same_hemisphere(wo, wi) & any(wh != 0.f);
    wh = normalize(wh);
    auto D = _distribution->D(wh);
    auto absCosThetaI = abs_cos_theta(wi);
    auto absCosThetaO = abs_cos_theta(wo);
    auto pow5 = [](auto &&v) noexcept { return sqr(sqr(v)) * v; };
    auto diffuse = (28.f / (23.f * pi)) * _rd * (1.f - _rs) *
                   (1.f - pow5(1.f - .5f * absCosThetaI)) *
                   (1.f - pow5(1.f - .5f * absCosThetaO));
    auto specular = D / (4.f * abs_dot(wi, wh) * max(absCosThetaI, absCosThetaO)) *
                    Schlick(dot(wi, wh));
    return ite(valid, diffuse + specular, 0.f);
}

BxDF::SampledDirection FresnelBlend::sample_wi(Expr<float3> wo, Expr<float2> uOrig, TransportMode mode) const noexcept {
    auto u = def(uOrig);
    auto wi = def(make_float3());
    $if(u.x < _rd_ratio) {
        u.x = u.x / _rd_ratio;
        // Cosine-sample the hemisphere, flipping the direction if necessary
        wi = sample_cosine_hemisphere(u);
        wi.z *= sign(cos_theta(wo));
    }
    $else {
        u.x = (u.x - _rd_ratio) / (1.f - _rd_ratio);
        // Sample microfacet orientation $\wh$ and reflected direction $\wi$
        auto wh = _distribution->sample_wh(wo, u);
        wi = reflect(-wo, wh);
    };
    return {.wi = wi, .valid = same_hemisphere(wo, wi)};
}

Float FresnelBlend::pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    auto wh = normalize(wo + wi);
    auto pdf_wh = _distribution->pdf(wo, wh);
    auto p = lerp(pdf_wh / (4.f * dot(wo, wh)), abs_cos_theta(wi) * inv_pi, _rd_ratio);
    return ite(same_hemisphere(wo, wi), p, 0.f);
}

FresnelBlend::Gradient FresnelBlend::backward(
    Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode) const noexcept {

    auto pow5 = [](auto &&v) noexcept { return sqr(sqr(v)) * v; };
    auto absCosThetaI = abs_cos_theta(wi);
    auto absCosThetaO = abs_cos_theta(wo);

    auto wh = wi + wo;
    auto valid = same_hemisphere(wo, wi) & any(wh != 0.f);
    wh = normalize(wh);

    auto k = (28.f / (23.f * pi)) *
             (1.f - pow5(1.f - .5f * absCosThetaI)) *
             (1.f - pow5(1.f - .5f * absCosThetaO));
    auto dv = df * ite(valid, 1.f, 0.f);
    auto diffuse_rd = (1.f - _rs) * k;
    auto diffuse_rs = -_rd * k;
    auto specular_rs = _distribution->D(wh) /
                       (4.f * abs_dot(wi, wh) * max(absCosThetaI, absCosThetaO)) *
                       (1 - pow5(1.f - dot(wi, wh)));

    auto d_rd = dv * (diffuse_rd);
    auto d_rs = dv * (diffuse_rs + specular_rs);
    auto d_alpha = (dv * Schlick(dot(wi, wh))).sum() * _distribution->grad_D(wh).dAlpha /
                   (4.f * abs_dot(wi, wh) * max(absCosThetaI, absCosThetaO));
    return {.dRd = d_rd, .dRs = d_rs, .dAlpha = d_alpha};
}

}// namespace luisa::render

//
// Created by Mike Smith on 2022/1/31.
//

#include <dsl/sugar.h>
#include <util/frame.h>
#include <util/sampling.h>
#include <util/scattering.h>

namespace luisa::render {

Bool refract(Float3 wi, Float3 n, Float eta, Float3 *wt) noexcept {

    using namespace compute;

    // Compute $\cos \theta_\roman{t}$ using Snell's law
    auto cosThetaI = dot(n, wi);
    auto sin2ThetaI = max(0.0f, 1.0f - sqr(cosThetaI));
    auto sin2ThetaT = sqr(eta) * sin2ThetaI;
    auto cosThetaT = sqrt(saturate(1.f - sin2ThetaT));
    *wt = -eta * wi + (eta * cosThetaI - cosThetaT) * n;
    // Handle total internal reflection for transmission
    return sin2ThetaT < 1.f;
}

Float3 reflect(Float3 wo, Float3 n) noexcept {
    using compute::dot;
    return -wo + 2.0f * dot(wo, n) * n;
}

template<typename T>
[[nodiscard]] inline auto fresnel_dielectric_impl(Float cosThetaI_in, T etaI_in, T etaT_in) noexcept {
    using namespace compute;
    auto cosThetaI = clamp(cosThetaI_in, -1.f, 1.f);
    // Potentially swap indices of refraction
    auto entering = cosThetaI > 0.f;
    auto etaI = ite(entering, etaI_in, etaT_in);
    auto etaT = ite(entering, etaT_in, etaI_in);
    cosThetaI = abs(cosThetaI);
    // Compute _cosThetaT_ using Snell's law
    auto sinThetaI = sqrt(max(0.f, 1.f - sqr(cosThetaI)));
    auto sinThetaT = etaI / etaT * sinThetaI;
    auto cosThetaT = sqrt(max(0.f, 1.f - sqr(sinThetaT)));
    auto Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                 ((etaT * cosThetaI) + (etaI * cosThetaT));
    auto Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                 ((etaI * cosThetaI) + (etaT * cosThetaT));
    auto fr = (Rparl * Rparl + Rperp * Rperp) * .5f;
    // Handle total internal reflection
    return ite(sinThetaT < 1.f, fr, 1.f);
}

Float fresnel_dielectric(Float cosThetaI, Float etaI, Float etaT) noexcept {
    return fresnel_dielectric_impl(std::move(cosThetaI), std::move(etaI), std::move(etaT));
}

Float4 fresnel_dielectric(Float cosThetaI, Float4 etaI, Float4 etaT) noexcept {
    return fresnel_dielectric_impl(std::move(cosThetaI), std::move(etaI), std::move(etaT));
}

Float4 fresnel_conductor(Float cosThetaI, Float4 etai, Float4 etat, Float4 k) noexcept {
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
    return sign(dot(v, n)) * v;
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

Float MicrofacetDistribution::G1(Expr<float3> w) const noexcept {
    return 1.0f / (1.0f + Lambda(w));
}

Float MicrofacetDistribution::G(Expr<float3> wo, Expr<float3> wi) const noexcept {
    return 1.0f / (1.0f + Lambda(wo) + Lambda(wi));
}

Float MicrofacetDistribution::pdf(Expr<float3> wo, Expr<float3> wh) const noexcept {
    using compute::abs;
    using compute::dot;
    return D(wh) * G1(wo) * abs_dot(wo, wh) / abs_cos_theta(wo);
}

TrowbridgeReitzDistribution::TrowbridgeReitzDistribution(Expr<float2> alpha) noexcept
    : _alpha{compute::max(alpha, 1e-3f)} {}

Float TrowbridgeReitzDistribution::roughness_to_alpha(Expr<float> roughness) noexcept { return sqr(roughness); }
Float2 TrowbridgeReitzDistribution::roughness_to_alpha(Expr<float2> roughness) noexcept {
    return compute::make_float2(
        roughness_to_alpha(roughness.x),
        roughness_to_alpha(roughness.y));
}

Float TrowbridgeReitzDistribution::D(Expr<float3> wh) const noexcept {
    using compute::isinf;
    auto tan2Theta = tan2_theta(wh);
    auto cos4Theta = sqr(cos2_theta(wh));
    auto e = tan2Theta * (sqr(cos_phi(wh) / _alpha.x) +
                          sqr(sin_phi(wh) / _alpha.y));
    auto d = 1.0f / (pi * _alpha.x * _alpha.y * cos4Theta * sqr(1.f + e));
    return ite(isinf(tan2Theta), 0.f, d);
}

Float TrowbridgeReitzDistribution::Lambda(Expr<float3> w) const noexcept {
    using compute::isinf;
    auto absTanTheta = abs(tan_theta(w));
    // Compute _alpha_ for direction _w_
    auto alpha2 = cos2_phi(w) * sqr(_alpha.x) +
                  sin2_phi(w) * sqr(_alpha.y);
    auto alpha2Tan2Theta = alpha2 * sqr(absTanTheta);
    auto L = (-1.f + sqrt(1.f + alpha2Tan2Theta)) * .5f;
    return ite(isinf(absTanTheta), 0.f, L);
}

[[nodiscard]] inline Float2 TrowbridgeReitzSample11(Float cosTheta, Float2 U) noexcept {

    using namespace luisa::compute;

    // special case (normal incidence)
    auto r = sqrt(U.x / (1.f - U.x));
    auto phi = (2.f * pi) * U.y;
    auto special_slope = r * make_float2(cos(phi), sin(phi));
    auto sinTheta = sqrt(saturate(1.f - sqr(cosTheta)));
    auto tanTheta = sinTheta / cosTheta;
    auto a = 1.f / tanTheta;
    auto G1 = 2.f / (1.f + sqrt(1.f + 1.f / sqr(a)));

    // sample slope_x
    auto A = 2.f * U.x / G1 - 1.f;
    auto tmp = min(1.f / (sqr(A) - 1.f), 1e10f);
    auto B = tanTheta;
    auto D = sqrt(max(sqr(B) * sqr(tmp) - (sqr(A) - sqr(B)) * tmp, 0.f));
    auto slope_x_1 = B * tmp - D;
    auto slope_x_2 = B * tmp + D;
    auto slope_x = ite(
        (A < 0.f) | (slope_x_2 > 1.f / tanTheta),
        slope_x_1, slope_x_2);

    // sample slope_y
    auto S = ite(U.y > .5f, 1.f, -1.f);
    auto U2 = ite(U.y > .5f, 2.f * (U.y - .5f), 2.f * (.5f - U.y));
    auto z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
             (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    auto slope_y = S * z * sqrt(1.f + sqr(slope_x));
    auto slope = make_float2(slope_x, slope_y);
    return ite(cosTheta > .9999f, special_slope, slope);
}

[[nodiscard]] inline Float3 TrowbridgeReitzSample(Float3 wi, Float2 alpha, Float2 U) noexcept {

    using compute::make_float2;
    using compute::make_float3;
    using compute::normalize;

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
}

Float3 TrowbridgeReitzDistribution::sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept {
    auto s = sign(cos_theta(wo));
    auto wh = TrowbridgeReitzSample(s * wo, _alpha, u);
    return s * wh;
}

Float4 FresnelConductor::evaluate(Expr<float> cosThetaI) const noexcept {
    return fresnel_conductor(abs(cosThetaI), _eta_i, _eta_t, _k);
}

luisa::map<luisa::string, Float4> FresnelConductor::grad(Expr<float> cosThetaI) const noexcept {
    LUISA_ERROR_WITH_LOCATION("FresnelConductor is not differentiable.");
}
bool FresnelConductor::differentiable() const noexcept {
    return false;
}

Float4 FresnelDielectric::evaluate(Expr<float> cosThetaI) const noexcept {
    return fresnel_dielectric(cosThetaI, _eta_i, _eta_t);
}

luisa::map<luisa::string, Float4> FresnelDielectric::grad(Expr<float> cosThetaI_in) const noexcept {
    using namespace compute;

    auto cosThetaI = clamp(cosThetaI_in, -1.f, 1.f);

    // Potentially swap indices of refraction
    auto entering = cosThetaI > 0.f;
    auto etaI = ite(entering, _eta_i, _eta_t);
    auto etaT = ite(entering, _eta_t, _eta_i);

    cosThetaI = abs(cosThetaI);
    auto sinThetaI = sqrt(max(0.f, 1.f - sqr(cosThetaI)));

    // Compute _cosThetaT_ using Snell's law
    auto sinThetaT = sinThetaI * etaI / etaT;
    auto cos2ThetaT = 1.f - sqr(sinThetaT);
    auto cosThetaT = sqrt(max(0.f, cos2ThetaT));

    auto Rparlu = (etaT * cosThetaI) - (etaI * cosThetaT);
    auto Rparlv = (etaT * cosThetaI) + (etaI * cosThetaT);
    auto Rparl = Rparlu / Rparlv;
    auto Rperpu = (etaI * cosThetaI) - (etaT * cosThetaT);
    auto Rperpv = (etaI * cosThetaI) + (etaT * cosThetaT);
    auto Rperp = Rperpu / Rperpv;

    //    // forward
    //    auto fr_Rparl = Rparl;
    //    auto fr_Rperp = Rperp;
    //    auto Rparlu_etaI = -cosThetaT;
    //    auto Rparlv_etaI = cosThetaT;
    //    auto Rparlu_etaT = cosThetaI;
    //    auto Rparlv_etaT = cosThetaI;
    //    auto Rparl_etaI = (Rparlu_etaI * Rparlv - Rparlu * Rparlv_etaI) / sqr(Rparlv);
    //    auto Rparl_etaT = (Rparlu_etaT * Rparlv - Rparlu * Rparlv_etaT) / sqr(Rparlv);
    //    auto Rperpu_etaI = cosThetaI;
    //    auto Rperpv_etaI = cosThetaI;
    //    auto Rperpu_etaT = -cosThetaT;
    //    auto Rperpv_etaT = cosThetaT;
    //    auto Rperp_etaI = (Rperpu_etaI * Rperpv - Rperpv_etaI * Rperpu) / sqr(Rperpv);
    //    auto Rperp_etaT = (Rperpu_etaT * Rperpv - Rperpv_etaT * Rperpu) / sqr(Rperpv);
    //    auto fr_etaI = fr_Rparl * Rparl_etaI + fr_Rperp * Rperp_etaI;
    //    auto fr_etaT = fr_Rparl * Rparl_etaT + fr_Rperp * Rperp_etaT;
    //    auto not_tir = sinThetaT < 1.f;
    //    auto f_etaI = ite(not_tir, ite(entering, fr_etaI, fr_etaT), 0.f);
    //    auto f_etaT = ite(not_tir, ite(entering, fr_etaT, fr_etaI), 0.f);
    //    return {{"f_etaI", f_etaI}, {"f_etaT", f_etaT}};

    // backward
    auto d_fr = ite(sinThetaT < 1.f, 1.f, 0.f);
    auto d_Rparl = d_fr * Rparl;
    auto d_Rperp = d_fr * Rperp;
    auto d_Rparlu = d_Rparl / Rparlv;
    auto d_Rparlv = -d_Rparl * Rparlu / (Rparlv * Rparlv);
    auto d_Rperpu = d_Rperp / Rperpv;
    auto d_Rperpv = -d_Rperp * Rperpu / (Rperpv * Rperpv);
    auto d_cosThetaT = (-d_Rparlu + d_Rparlu) * etaT + (-d_Rperpu + d_Rperpv) * etaI;
    auto d_sinThetaT = ite(cos2ThetaT > 0.f, -d_cosThetaT * sinThetaT / sqrt(cos2ThetaT), 0.f);
    auto d_etaT = -d_sinThetaT * sinThetaI * etaI / (etaT * etaT) +
                  (d_Rparlu + d_Rparlv) * cosThetaI + (-d_Rperpu + d_Rperpv) * cosThetaT;
    auto d_etaI = d_sinThetaT * sinThetaI / etaT +
                  (-d_Rparlu + d_Rparlv) * cosThetaT + (d_Rperpu + d_Rperpv) * cosThetaI;
    auto d_eta_i =
        d_etaI * ite(entering, 1.f, 0.f) +
        d_etaT * ite(entering, 0.f, 1.f);
    auto d_eta_t =
        d_etaI * ite(entering, 0.f, 1.f) +
        d_etaT * ite(entering, 1.f, 0.f);

    return {{"d_eta_i", d_eta_i}, {"d_eta_t", d_eta_t}};
}
bool FresnelDielectric::differentiable() const noexcept {
    return true;
}

Float4 FresnelNoOp::evaluate(Expr<float>) const noexcept {
    return make_float4(1.0f);
}

luisa::map<luisa::string, Float4> FresnelNoOp::grad(Expr<float> cosThetaI) const noexcept {
    LUISA_ERROR_WITH_LOCATION("FresnelNoOp is not differentiable.");
}
bool FresnelNoOp::differentiable() const noexcept {
    return false;
}

Float4 BxDF::sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *p) const noexcept {
    *wi = sample_cosine_hemisphere(u);
    wi->z *= compute::sign(cos_theta(wo));
    *p = pdf(wo, *wi);
    return evaluate(wo, *wi);
}

Float BxDF::pdf(Expr<float3> wo, Expr<float3> wi) const noexcept {
    return compute::ite(same_hemisphere(wo, wi), abs_cos_theta(wi) * inv_pi, 0.0f);
}

Float4 LambertianReflection::evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
    return _r * inv_pi;
}
luisa::map<luisa::string, Float4> LambertianReflection::grad(Expr<float3> wo, Expr<float3> wi) const noexcept {
    auto d_r = make_float4(inv_pi);
    return {{"d_r", d_r}};
}

Float4 LambertianTransmission::evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
    return _t * inv_pi;
}

Float4 LambertianTransmission::sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *p) const noexcept {
    *wi = sample_cosine_hemisphere(u);
    wi->z *= -compute::sign(cos_theta(wo));
    *p = pdf(wo, *wi);
    return evaluate(wo, *wi);
}

Float LambertianTransmission::pdf(Expr<float3> wo, Expr<float3> wi) const noexcept {
    return compute::ite(same_hemisphere(wo, wi), 0.0f, abs_cos_theta(wi) * inv_pi);
}
luisa::map<luisa::string, Float4> LambertianTransmission::grad(Expr<float3> wo, Expr<float3> wi) const noexcept {
    auto d_t = make_float4(inv_pi);
    return {{"d_t", d_t}};
}

Float4 MicrofacetReflection::evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
    using compute::any;
    using compute::normalize;
    auto cosThetaO = abs_cos_theta(wo);
    auto cosThetaI = abs_cos_theta(wi);
    auto wh = wi + wo;
    auto valid = cosThetaI != 0.f & cosThetaO != 0.f & any(wh != 0.f);
    wh = normalize(wh);
    // For the Fresnel call, make sure that wh is in the same hemisphere
    // as the surface normal, so that TIR is handled correctly.
    auto F = _fresnel->evaluate(dot(wi, face_forward(wh, make_float3(0.f, 0.f, 1.f))));
    auto D = _distribution->D(wh);
    auto G = _distribution->G(wo, wi);
    auto f = 0.25f * _r * D * G * F / (cosThetaI * cosThetaO);
    return ite(valid, f, 0.f);
}

Float4 MicrofacetReflection::sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *p) const noexcept {
    // Sample microfacet orientation $\wh$ and reflected direction $\wi$
    *p = 0.0f;
    auto wh = _distribution->sample_wh(wo, u);
    *wi = reflect(wo, wh);
    auto valid = wo.z != 0.f & same_hemisphere(wo, wh) & same_hemisphere(wo, *wi);
    // Compute PDF of _wi_ for microfacet reflection
    *p = ite(valid, _distribution->pdf(wo, wh) / (4.f * dot(wo, wh)), 0.f);
    return ite(valid, evaluate(wo, *wi), 0.f);
}

Float MicrofacetReflection::pdf(Expr<float3> wo, Expr<float3> wi) const noexcept {
    auto wh = normalize(wo + wi);
    auto p = _distribution->pdf(wo, wh) / (4.f * dot(wo, wh));
    return ite(same_hemisphere(wo, wi), p, 0.f);
}
luisa::map<luisa::string, Float4> MicrofacetReflection::grad(Expr<float3> wo, Expr<float3> wi) const noexcept {
    // TODO : we didn't deal with distribution here

    using compute::any;
    using compute::normalize;
    auto cosThetaO = abs_cos_theta(wo);
    auto cosThetaI = abs_cos_theta(wi);
    auto wh = wi + wo;
    auto valid = cosThetaI != 0.f & cosThetaO != 0.f & any(wh != 0.f);
    wh = normalize(wh);
    auto cosI = dot(wi, face_forward(wh, make_float3(0.f, 0.f, 1.f)));
    auto F = _fresnel->evaluate(cosI);
    auto D = _distribution->D(wh);
    auto G = _distribution->G(wo, wi);

    // backward
    luisa::map<luisa::string, Float4> grad_fresnel;
    if (_fresnel->differentiable())
        grad_fresnel = _fresnel->grad(cosI);
    auto d_f = ite(valid, 1.f, 0.f);
    auto d_r = d_f * 0.25f * D * G * F / (cosThetaI * cosThetaO);
    auto d_F = d_f * 0.25f * _r * D * G / (cosThetaI * cosThetaO);

    luisa::map<luisa::string, Float4> grad;
    grad["d_r"] = d_r;
    for (const auto &v : grad_fresnel) {
        grad[v.first] = d_F * v.second;
    }

    return grad;
}

Float4 MicrofacetTransmission::evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
    auto cosThetaO = cos_theta(wo);
    auto cosThetaI = cos_theta(wi);
    auto refr = !same_hemisphere(wo, wi) & cosThetaO != 0.f & cosThetaI != 0.f;
    auto eta = ite(cosThetaO > 0.f, _eta_b / _eta_a, _eta_a / _eta_b);
    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    auto G = _distribution->G(wo, wi);
    auto ft = def(make_float4());
    for (auto i = 0u; i < 4u; i++) {
        auto wh = normalize(wo + wi * eta[i]);
        wh = compute::sign(cos_theta(wh)) * wh;
        auto sqrtDenom = dot(wo, wh) + eta[i] * dot(wi, wh);
        auto factor = 1.f / eta[i];
        auto F = _fresnel.evaluate(dot(wo, wh))[i];
        auto D = _distribution->D(wh);
        auto f = (1.f - F) * _t[i] * sqr(factor) *
                 abs(D * G * sqr(eta[i]) * abs_dot(wi, wh) * abs_dot(wo, wh) /
                     (cosThetaI * cosThetaO * sqr(sqrtDenom)));
        auto valid = refr & dot(wo, wh) * dot(wi, wh) < 0.f;
        ft[i] = ite(valid, f, 0.f);
    }
    return ft;
}

Float4 MicrofacetTransmission::sample(Expr<float3> wo, Float3 *wi, Expr<float2> u_in, Float *p) const noexcept {
    using namespace compute;
    auto swl_i_float = u_in.x * 4.f;
    auto swl_i = cast<int>(clamp(swl_i_float, 0.f, 3.f));
    auto eta = ite(cos_theta(wo) > 0.f, _eta_a / _eta_b, _eta_b / _eta_a)[swl_i];// TODO
    auto u = make_float2(fract(swl_i_float), u_in.y);
    auto wh = _distribution->sample_wh(wo, u);
    auto refr = refract(wo, wh, eta, wi);
    *p = pdf(wo, *wi);
    return evaluate(wo, *wi);
}

Float MicrofacetTransmission::pdf(Expr<float3> wo, Expr<float3> wi) const noexcept {
    auto pdf = def(0.f);
    auto eta = ite(cos_theta(wo) > 0.f, _eta_b / _eta_a, _eta_a / _eta_b);
    for (auto i = 0u; i < 4u; i++) {
        auto wh = normalize(wo + wi * eta[i]);
        // Compute change of variables _dwh\_dwi_ for microfacet transmission
        auto sqrtDenom = dot(wo, wh) + eta[i] * dot(wi, wh);
        auto dwh_dwi = sqr(eta[i] / sqrtDenom) * abs_dot(wi, wh);
        auto valid = !same_hemisphere(wo, wi) & dot(wo, wh) * dot(wi, wh) < 0.f;
        pdf += ite(valid, _distribution->pdf(wo, wh) * dwh_dwi, 0.f);
    }
    return pdf * .25f;
}

luisa::map<luisa::string, Float4> MicrofacetTransmission::grad(Expr<float3> wo, Expr<float3> wi) const noexcept {
    // TODO
    LUISA_ERROR_WITH_LOCATION("unimplemented");

    auto cosThetaO = cos_theta(wo);
    auto cosThetaI = cos_theta(wi);
    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    auto eta = ite(cosThetaO > 0.f, _eta_b / _eta_a, _eta_a / _eta_b)[0];// TODO
    auto wh = normalize(wo + wi * eta);
    wh = compute::sign(cos_theta(wh)) * wh;
    auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
    auto factor = 1.f / eta;
    auto F = _fresnel.evaluate(dot(wo, wh));
    auto D = _distribution->D(wh);
    auto G = _distribution->G(wo, wi);
    auto f = (1.f - F) * _t * sqr(factor) *
             abs(D * G * sqr(eta) * abs_dot(wi, wh) * abs_dot(wo, wh) /
                 (cosThetaI * cosThetaO * sqr(sqrtDenom)));
    auto valid = !same_hemisphere(wo, wi) &
                 cosThetaO != 0.f & cosThetaI != 0.f &
                 dot(wo, wh) * dot(wi, wh) < 0.f;

    // backward
    auto k_0 = abs(D * G * sqr(eta) * dot(wi, wh) * dot(wo, wh) /
                   (cosThetaI * cosThetaO * sqr(sqrtDenom)));
    auto d_f = ite(valid, 1.f, 0.f);
    auto d_t = d_f * (1.f - F) * sqr(factor) * k_0;
    auto d_F = -d_f * _t * sqr(factor) * k_0;
    auto d_factor = d_f * (1.f - F) * _t * k_0 * 2.f * factor;
    auto d_sqrtDenom = d_f * (1.f - F) * _t * sqr(factor) * k_0 / sqrtDenom * (-2.f);
    //    auto d_wh = d_f * (1.f - F) * _t * sqr(factor) * D * G * sqr(eta) *
    //                abs(D * G * sqr(eta) *
    //                    (wi * dot(wo, wh) + abs_dot(wi, wh) * wo) /
    //                    (cosThetaI * cosThetaO * sqr(sqrtDenom)));
    auto d_eta = -d_factor / sqr(eta);// TODO
}

OrenNayar::OrenNayar(Expr<float4> R, Expr<float> sigma) noexcept
    : _r{R}, _sigma{sigma} {
    auto sigma2 = sqr(radians(sigma));
    _a = 1.f - (sigma2 / (2.f * sigma2 + 0.66f));
    _b = 0.45f * sigma2 / (sigma2 + 0.09f);
}

Float4 OrenNayar::evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
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
    return _r * inv_pi * (_a + _b * maxCos * sinAlpha * tanBeta);
}

luisa::map<luisa::string, Float4> OrenNayar::grad(Expr<float3> wo, Expr<float3> wi) const noexcept {
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

    auto sigma2 = sqr(radians(_sigma));

    //    // forward
    //    auto sigma2_sigma = 2 * radians(_sigma) / 180.f;
    //    auto a_sigma2 = -0.165f * sqr(sigma2 + 0.33f);
    //    auto b_sigma2 = 0.0405f / sqr(sigma2 + 0.09f);
    //
    //    auto f_r = make_float4(inv_pi * (_a + _b * maxCos * sinAlpha * tanBeta));
    //    auto f_a = _r * inv_pi;
    //    auto f_b = _r * inv_pi * maxCos * sinAlpha * tanBeta;
    //    auto f_sigma = (f_a * a_sigma2 + f_b * b_sigma2) * sigma2_sigma;
    //
    //    return {f_r, f_sigma};

    // backward
    auto sigma2_sigma = 2 * radians(_sigma) / 180.f;
    auto a_sigma2 = -0.165f * sqr(sigma2 + 0.33f);
    auto b_sigma2 = 0.0405f / sqr(sigma2 + 0.09f);
    auto d_r = make_float4(inv_pi * (_a + _b * maxCos * sinAlpha * tanBeta));
    auto d_a = _r * inv_pi;
    auto d_b = _r * inv_pi * maxCos * sinAlpha * tanBeta;
    auto d_sigma2 = d_a * a_sigma2 + d_b * b_sigma2;
    auto d_sigma = d_sigma2 * sigma2_sigma;

    return {{"d_r", d_r}, {"d_sigma", d_sigma}};
}

Float4 FresnelBlend::Schlick(Expr<float> cosTheta) const noexcept {
    auto pow5 = [](Float v) { return sqr(sqr(v)) * v; };
    return _rs + pow5(1.f - cosTheta) * (1.f - _rs);
}

Float4 FresnelBlend::evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
    auto pow5 = [](Float v) noexcept { return sqr(sqr(v)) * v; };
    auto absCosThetaI = abs_cos_theta(wi);
    auto absCosThetaO = abs_cos_theta(wo);
    auto diffuse = (28.f / (23.f * pi)) * _rd * (1.f - _rs) *
                   (1.f - pow5(1.f - .5f * absCosThetaI)) *
                   (1.f - pow5(1.f - .5f * absCosThetaO));
    auto wh = wi + wo;
    auto valid = any(wh != 0.f);
    wh = normalize(wh);
    auto specular = _distribution->D(wh) /
                    (4.f * abs_dot(wi, wh) * max(absCosThetaI, absCosThetaO)) *
                    Schlick(dot(wi, wh));
    return ite(valid, diffuse + specular, 0.f);
}

Float4 FresnelBlend::sample(Expr<float3> wo, Float3 *wi, Expr<float2> uOrig, Float *p) const noexcept {
    using compute::sign;
    auto u = def(uOrig);
    *p = 0.f;
    auto f = def(make_float4());
    $if(u.x < .5f) {
        u.x = 2.f * u.x;
        // Cosine-sample the hemisphere, flipping the direction if necessary
        *wi = sample_cosine_hemisphere(u);
        wi->z *= sign(cos_theta(wo));
        compute::assume(same_hemisphere(wo, *wi));
        *p = pdf(wo, *wi);
        f = evaluate(wo, *wi);
    }
    $else {
        u.x = 2.f * (u.x - .5f);
        // Sample microfacet orientation $\wh$ and reflected direction $\wi$
        auto wh = _distribution->sample_wh(wo, u);
        *wi = reflect(wo, wh);
        $if(same_hemisphere(wo, *wi)) {
            *p = pdf(wo, *wi);
            f = evaluate(wo, *wi);
        };
    };
    return f;
}

Float FresnelBlend::pdf(Expr<float3> wo, Expr<float3> wi) const noexcept {
    auto wh = normalize(wo + wi);
    auto pdf_wh = _distribution->pdf(wo, wh);
    auto p = .5f * (abs_cos_theta(wi) * inv_pi + pdf_wh / (4.f * dot(wo, wh)));
    return ite(same_hemisphere(wo, wi), p, 0.f);
}
luisa::map<luisa::string, Float4> FresnelBlend::grad(Expr<float3> wo, Expr<float3> wi) const noexcept {
    // TODO : we didn't deal with distribution here

    auto pow5 = [](Float v) noexcept { return sqr(sqr(v)) * v; };
    auto absCosThetaI = abs_cos_theta(wi);
    auto absCosThetaO = abs_cos_theta(wo);

    auto wh = wi + wo;
    auto valid = any(wh != 0.f);
    wh = normalize(wh);

    auto diffuse_rd = (28.f / (23.f * pi)) * (1.f - _rs) *
                      (1.f - pow5(1.f - .5f * absCosThetaI)) *
                      (1.f - pow5(1.f - .5f * absCosThetaO));
    auto diffuse_rs = -(28.f / (23.f * pi)) * _rd *
                      (1.f - pow5(1.f - .5f * absCosThetaI)) *
                      (1.f - pow5(1.f - .5f * absCosThetaO));
    auto specular_rs = 1 - pow5(1.f - dot(wi, wh));

    auto d_rd = make_float4(diffuse_rd);
    auto d_rs = make_float4(diffuse_rs + specular_rs);
    return {{"d_rd", d_rd}, {"d_rs", d_rs}};
}

}// namespace luisa::render

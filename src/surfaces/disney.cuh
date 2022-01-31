#include <cstdio>

#include "material/material.h"
#include "misc/shader_helpers.h"
#include "misc/sampling.h"
#include "misc/payload.h"
#include "misc/intersection.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <stdint.h>

#include <sutil/vec_math.h>

namespace mtl_disney {

namespace helpers {

[[nodiscard]] inline Float sqr(Float x) { return x * x; }

template<typename X, typename T>
[[nodiscard]] inline auto Lerp(X t, T v1, T v2) { return (1.0f - t) * v1 + t * v2; }

[[nodiscard]] inline Float AbsCosTheta(Float3 w) { return fabsf(w.z); }

[[nodiscard]] inline bool SameHemisphere(Float3 w, Float3 wp) {
    return w.z * wp.z > 0;
}

[[nodiscard]] inline Float3 Faceforward(Float3 v, Float3 v2) {
    return (dot(v, v2) < 0.0f) ? -v : v;
}

[[nodiscard]] inline Float3 SphericalDirection(Float sinTheta, Float cosTheta, Float phi) {
    return make_Float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

[[nodiscard]] inline Float3 Reflect(Float3 wo, Float3 n) {
    return -wo + 2.0f * dot(wo, n) * n;
}

[[nodiscard]] inline bool Refract(Float3 wi, Float3 n, Float eta, Float3* wt) {
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    auto cosThetaI = dot(n, wi);
    auto sin2ThetaI = fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
    auto sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1.0f) return false;
    auto cosThetaT = sqrtf(1.0f - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

// F = F0 + (1- F0)(1- cos)^5

[[nodiscard]] inline Float SchlickWeight(Float cosTheta) {  // (1 - cos)^5
    Float m = clamp(1.0f - cosTheta, 0.0f, 1.0f);
    return (m * m) * (m * m) * m;
}

[[nodiscard]] inline Float FrSchlick(Float R0, Float cosTheta) {
    return Lerp(SchlickWeight(cosTheta), R0, 1.0f);
}

[[nodiscard]] inline auto FrSchlick(Float3 R0, Float cosTheta) {
    return Lerp(SchlickWeight(cosTheta), R0, make_Float3(1.0f, 1.0f, 1.0f));
}

[[nodiscard]] inline Float FresnelMoment1(Float eta) {
    auto eta2 = eta * eta;
    auto eta3 = eta2 * eta;
    auto eta4 = eta3 * eta;
    auto eta5 = eta4 * eta;
    if (eta < 1.0f) {
        return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945f * eta3 +
               2.49277f * eta4 - 0.68441f * eta5;
    }
    return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
           1.27198f * eta4 + 0.12746f * eta5;
}

[[nodiscard]] inline Float FrDielectric(Float cosThetaI, Float etaI, Float etaT) {
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        auto temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = fabsf(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    auto sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    auto sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1.0f) { return 1.0f; }
    auto cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    auto Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    auto Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

// For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
// coming from air..
[[nodiscard]] inline Float SchlickR0FromEta(Float eta) { return sqr(eta - 1.0f) / sqr(eta + 1.0f); }

[[nodiscard]] inline Float GTR1(Float cosTheta, Float alpha) {
    auto alpha2 = alpha * alpha;
    return (alpha2 - 1.0f) / (M_PIf * logf(alpha2) * (1.0f + (alpha2 - 1.0f) * cosTheta * cosTheta));
}

// Smith masking/shadowing term.
[[nodiscard]] inline Float smithG_GGX(Float cosTheta, Float alpha) {
    auto alpha2 = alpha * alpha;
    auto cosTheta2 = cosTheta * cosTheta;
    return 1.0f / (cosTheta + sqrtf(alpha2 + cosTheta2 - alpha2 * cosTheta2));
}

[[nodiscard]] inline Float AbsDot(Float3 u, Float3 v) noexcept {
    return fabs(dot(u, v));
}

}

namespace disney {  // Definitions of DisneyMaterial layers, all in **LOCAL** coordinate space. Credit to pbrt-v3.

using namespace helpers;

template<typename T>
struct BxDF {

    [[nodiscard]] Float pdf(Float3 wo, Float3 wi) const noexcept {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * M_1_PIf : 0.0f;
    }

    [[nodiscard]] Float3 sample_f(Float3 wo, Float3* wi, Float2 u, Float* p) const noexcept {
        auto w = gr::cosineSampleHemisphere(u.x, u.y);
        *wi = w;
        *p = w.z * M_1_PIf;
        return static_cast<const T*>(this)->f(wo, w);
    }

};

class Diffuse : public BxDF<Diffuse> {

private:
    Float3 R;

public:
    [[nodiscard]] explicit Diffuse(Float3 R = make_Float3(0.0f)) noexcept : R{ R } {}

    [[nodiscard]] Float3 f(Float3 wo, Float3 wi) const noexcept {
        auto Fo = SchlickWeight(AbsCosTheta(wo));
        auto Fi = SchlickWeight(AbsCosTheta(wi));

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
        // Burley 2015, eq (4).
        return R * M_1_PIf * (1 - Fo / 2) * (1 - Fi / 2);
    }
};

class FakeSS : public BxDF<FakeSS> {

private:
    Float3 R;
    Float roughness;

public:
    [[nodiscard]] explicit FakeSS(Float3 R = make_Float3(0.0f), Float roughness = 0.0f) noexcept : R{ R }, roughness{ roughness } {}

    [[nodiscard]] Float3 f(Float3 wo, Float3 wi) const noexcept {
        auto wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0) {
            return make_Float3(0.0f);
        }

        wh = normalize(wh);
        auto cosThetaD = dot(wi, wh);

        // Fss90 used to "flatten" retroreflection based on roughness
        auto Fss90 = cosThetaD * cosThetaD * roughness;
        auto Fo = SchlickWeight(AbsCosTheta(wo));
        auto Fi = SchlickWeight(AbsCosTheta(wi));
        auto Fss = Lerp(Fo, 1.0f, Fss90) * Lerp(Fi, 1.0f, Fss90);
        // 1.25 scale is used to (roughly) preserve albedo
        Float ss = 1.25f * (Fss * (1.0f / (AbsCosTheta(wo) + AbsCosTheta(wi)) - 0.5f) + 0.5f);

        return R * M_1_PIf * ss;
    }
};

class Retro : public BxDF<Retro> {

private:
    Float3 R;
    Float roughness;

public:
    [[nodiscard]] explicit Retro(Float3 R = make_Float3(0.0f), Float roughness = 0.0f) noexcept : R{ R }, roughness{ roughness } {}

    [[nodiscard]] Float3 f(Float3 wo, Float3 wi) const noexcept {
        auto wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0) { return make_Float3(0.0f); }
        wh = normalize(wh);
        auto cosThetaD = dot(wi, wh);

        Float Fo = SchlickWeight(AbsCosTheta(wo)),
              Fi = SchlickWeight(AbsCosTheta(wi));
        Float Rr = 2 * roughness * cosThetaD * cosThetaD;

        // Burley 2015, eq (4).
        return R * M_1_PIf * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
    }
};

class Sheen : public BxDF<Sheen> {

private:
    Float3 R;

public:
    [[nodiscard]] explicit Sheen(Float3 R = make_Float3(0.0f)) noexcept : R{ R } {}

    [[nodiscard]] Float3 f(Float3 wo, Float3 wi) const noexcept {
        auto wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0) { return make_Float3(0.0f); }
        wh = normalize(wh);
        auto cosThetaD = dot(wi, wh);
        return R * SchlickWeight(cosThetaD);
    }
};

class Clearcoat : public BxDF<Clearcoat> {

private:
    Float weight;
    Float gloss;

public:
    [[nodiscard]] explicit Clearcoat(Float weight = 0.0f, Float gloss = 0.0f) noexcept : weight{ weight }, gloss{ gloss } {}

    [[nodiscard]] Float3 f(Float3 wo, Float3 wi) const noexcept {
        auto wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0) {
            return make_Float3(0.0f);
        }
        wh = normalize(wh);

        // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
        // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
        // (which is GTR2).
        auto Dr = GTR1(AbsCosTheta(wh), gloss);
        auto Fr = FrSchlick(.04f, dot(wo, wh));
        // The geometric term always based on alpha = 0.25.
        auto Gr = smithG_GGX(AbsCosTheta(wo), .25f) * smithG_GGX(AbsCosTheta(wi), .25f);

        return make_Float3(weight * Gr * Fr * Dr / 4.0f);
    }

    [[nodiscard]] [[nodiscard]] Float pdf(Float3 wo, Float3 wi) const noexcept {
        if (!SameHemisphere(wo, wi)) { return 0.0f; }

        auto wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0) { return 0.0f; }
        wh = normalize(wh);

        // The sampling routine samples wh exactly from the GTR1 distribution.
        // Thus, the final value of the PDF is just the value of the
        // distribution for wh converted to a mesure with respect to the
        // surface normal.
        auto Dr = GTR1(AbsCosTheta(wh), gloss);
        return Dr * AbsCosTheta(wh) / (4.0f * dot(wo, wh));
    }

    [[nodiscard]] Float3 sample_f(Float3 wo, Float3* wi, Float2 u, Float* p) const noexcept {

        // TODO: double check all this: there still seem to be some very
        // occasional fireflies with clearcoat; presumably there is a bug
        // somewhere.
        if (wo.z == 0.0f) { return make_Float3(0.0f); }

        auto alpha2 = gloss * gloss;
        auto cosTheta = sqrtf(fmaxf(0.0f, (1.0f - powf(alpha2, 1.0f - u.x)) / (1.0f - alpha2)));
        auto sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        auto phi = 2 * M_PIf * u.y;
        auto wh = SphericalDirection(sinTheta, cosTheta, phi);
        if (!SameHemisphere(wo, wh)) { wh = -wh; }

        auto w = Reflect(wo, wh);
        *wi = w;
        if (!SameHemisphere(wo, w)) { return make_Float3(0.f); }

        *p = pdf(wo, w);
        return f(wo, w);
    }
};

class Fresnel {

private:
    Float3 R0;
    Float metallic;
    Float eta;

public:
    [[nodiscard]] explicit Fresnel(Float3 R0 = make_Float3(0.0f), Float metallic = 0.0f, Float eta = 0.0f) noexcept : R0{ R0 }, metallic{ metallic }, eta{ eta } {}

    [[nodiscard]] Float3 evaluate(Float cosI) const {
        return Lerp(metallic, make_Float3(FrDielectric(cosI, 1.0f, eta)), FrSchlick(R0, cosI));
    }
};

[[nodiscard]] inline Float CosTheta(Float3 w) { return w.z; }
[[nodiscard]] inline Float Cos2Theta(Float3 w) { return w.z * w.z; }
[[nodiscard]] inline Float AbsCosTheta(Float3 w) { return fabs(w.z); }
[[nodiscard]] inline Float Sin2Theta(Float3 w) { return fmaxf(0.0f, 1.0f - Cos2Theta(w)); }
[[nodiscard]] inline Float SinTheta(Float3 w) { return sqrtf(Sin2Theta(w)); }
[[nodiscard]] inline Float TanTheta(Float3 w) { return SinTheta(w) / CosTheta(w); }
[[nodiscard]] inline Float Tan2Theta(Float3 w) { return Sin2Theta(w) / Cos2Theta(w); }
[[nodiscard]] inline Float CosPhi(Float3 w) { auto sinTheta = SinTheta(w); if (sinTheta == 0.0f) { return 1.0f; } else { return clamp(w.x / sinTheta, -1.0f, 1.0f); } }
[[nodiscard]] inline Float SinPhi(Float3 w) { auto sinTheta = SinTheta(w); if (sinTheta == 0.0f) { return 0.0f; } else { return clamp(w.y / sinTheta, -1.0f, 1.0f); } }
[[nodiscard]] inline Float Cos2Phi(Float3 w) { return CosPhi(w) * CosPhi(w); }
[[nodiscard]] inline Float Sin2Phi(Float3 w) { return SinPhi(w) * SinPhi(w); }

[[nodiscard]] inline Float2 TrowbridgeReitzSample11(Float cosTheta, Float U1, Float U2) {

    // special case (normal incidence)
    if (cosTheta > 0.9999f) {
        auto r = sqrtf(U1 / (1.0f - U1));
        auto phi = 6.28318530718f * U2;
        auto slope_x = r * cos(phi);
        auto slope_y = r * sin(phi);
        return make_Float2(slope_x, slope_y);
    }

    auto sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    auto tanTheta = sinTheta / cosTheta;
    auto a = 1.0f / tanTheta;
    auto G1 = 2.0f / (1.0f + sqrtf(1.f + 1.f / (a * a)));

    // sample slope_x
    auto A = 2.0f * U1 / G1 - 1.0f;
    auto tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10f) { tmp = 1e10f; }
    auto B = tanTheta;
    auto D = sqrtf(fmaxf(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.0f));
    auto slope_x_1 = B * tmp - D;
    auto slope_x_2 = B * tmp + D;
    auto slope_x = (A < 0.0f || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

    // sample slope_y
    auto S = 0.0f;
    if (U2 > 0.5f) {
        S = 1.f;
        U2 = 2.f * (U2 - .5f);
    }
    else {
        S = -1.f;
        U2 = 2.f * (.5f - U2);
    }
    auto z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
             (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    auto slope_y = S * z * sqrtf(1.f + slope_x * slope_x);

    return make_Float2(slope_x, slope_y);
}

[[nodiscard]] inline Float3 TrowbridgeReitzSample(Float3 wi, Float alpha_x, Float alpha_y, Float U1, Float U2) {
    // 1. stretch wi
    auto wiStretched = normalize(make_Float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    auto slope = TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2);
    auto slope_x = slope.x;
    auto slope_y = slope.y;

    // 3. rotate
    Float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    return normalize(make_Float3(-slope_x, -slope_y, 1.0f));
}

class TrowbridgeReitzDistribution {
public:
    // TrowbridgeReitzDistribution Public Methods
    static [[nodiscard]] inline Float RoughnessToAlpha(Float roughness) {
        roughness = fmaxf(roughness, 1e-3f);
        auto x = logf(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
    }

    [[nodiscard]] TrowbridgeReitzDistribution(Float alphax, Float alphay) : alphax(alphax), alphay(alphay) {}

    [[nodiscard]] Float G(Float3 wo, Float3 wi) const {
        return 1.0f / (1.0f + Lambda(wo) + Lambda(wi));
    }

    [[nodiscard]] [[nodiscard]] Float Pdf(Float3 wo, Float3 wh) const {
        return D(wh) * G1(wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
    }

    [[nodiscard]] Float D(Float3 wh) const {
        auto tan2Theta = Tan2Theta(wh);
        if (isinf(tan2Theta)) return 0.0f;
        auto cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
        auto e = (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) * tan2Theta;
        return 1.0f / (M_PIf * alphax * alphay * cos4Theta * (1.0f + e) * (1.0f + e));
    }

    [[nodiscard]] Float3 Sample_wh(Float3 wo, Float2 u) const {
        auto flip = wo.z < 0.0f;
        auto wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u.x, u.y);
        if (flip) wh = -wh;
        return wh;
    }

private:
    // TrowbridgeReitzDistribution Private Methods
    [[nodiscard]] Float Lambda(const Float3& w) const {
        auto absTanTheta = fabs(TanTheta(w));
        if (isinf(absTanTheta)) return 0.0f;
        // Compute _alpha_ for direction _w_
        auto alpha = sqrtf(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
        auto alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
        return (-1.0f + sqrtf(1.f + alpha2Tan2Theta)) * 0.5f;
    }

protected:
    [[nodiscard]] Float G1(Float3 w) const {
        //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
        return 1.0f / (1.0f + Lambda(w));
    }

    // TrowbridgeReitzDistribution Private Data
    Float alphax, alphay;
};

struct MicrofacetDistribution : TrowbridgeReitzDistribution {

    [[nodiscard]] explicit MicrofacetDistribution(Float alphax = 0.0f, Float alphay = 0.0f) : TrowbridgeReitzDistribution{ alphax, alphay } {}

    [[nodiscard]] Float G(Float3 wo, Float3 wi) const {
        // Disney uses the separable masking-shadowing model.
        return G1(wo) * G1(wi);
    }

};

template<typename Dist, typename Fres>
[[nodiscard]] Float3 MicrofacetReflection_f(Dist distrib, Fres fresnel, Float3 R, Float3 wo, Float3 wi) noexcept {
    Float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
    Float3 wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0.0f || cosThetaO == 0.0f) return make_Float3(0.0f);
    if (wh.x == 0.0f && wh.y == 0.0f && wh.z == 0.0f) return make_Float3(0.f);
    wh = normalize(wh);
    // For the Fresnel call, make sure that wh is in the same hemisphere
    // as the surface normal, so that TIR is handled correctly.
    auto F = fresnel.evaluate(dot(wi, Faceforward(wh, make_Float3(0.0f, 0.0f, 1.0f))));
    return R * distrib.D(wh) * distrib.G(wo, wi) * F / (4.0f * cosThetaI * cosThetaO);
}

template<typename Dist>
[[nodiscard]] Float MicrofacetReflection_pdf(Dist distrib, Float3 wo, const Float3& wi) noexcept {
    if (wo.z * wi.z <= 0.0f) return 0.0f;
    auto wh = normalize(wo + wi);
    return distrib.Pdf(wo, wh) / (4.0f * dot(wo, wh));
}

template<typename Dist, typename Fres>
[[nodiscard]] Float3 MicrofacetReflection_sample(Dist distrib, Fres fresnel, Float3 R, Float3 wo, Float3* wi, Float2 u, Float* pdf) noexcept {

    // Sample microfacet orientation $\wh$ and reflected direction $\wi$
    if (wo.z == 0.0f) return make_Float3(0.0f);
    auto wh = distrib.Sample_wh(wo, u);
    if (dot(wo, wh) < 0.0f) return make_Float3(0.0f);   // Should be rare
    *wi = Reflect(wo, wh);
    if (!SameHemisphere(wo, *wi)) return make_Float3(0.f);

    // Compute PDF of _wi_ for microfacet reflection
    *pdf = distrib.Pdf(wo, wh) / (4.0f * dot(wo, wh));
    return MicrofacetReflection_f(distrib, fresnel, R, wo, *wi);
}

[[nodiscard]] inline Float3 FresnelDielectric_f(Float etaI, Float etaT, Float cosThetaI) {
    return make_Float3(FrDielectric(cosThetaI, etaI, etaT));
}

template<typename Dist>
[[nodiscard]] Float3 MicrofacetTransmission_f(Dist distribution, Float3 T, Float etaA, Float etaB, Float3 wo, Float3 wi) {

    if (SameHemisphere(wo, wi)) return make_Float3(0.0f);  // transmission only

    auto cosThetaO = CosTheta(wo);
    auto cosThetaI = CosTheta(wi);
    if (cosThetaI == 0.0f || cosThetaO == 0.0f) return make_Float3(0.0f);

    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    auto eta = CosTheta(wo) > 0.0f ? (etaB / etaA) : (etaA / etaB);
    auto wh = normalize(wo + wi * eta);
    if (wh.z < 0.0f) wh = -wh;

    auto F = FresnelDielectric_f(etaA, etaB, dot(wo, wh));

    auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
    auto factor = 1.0f;

    return (make_Float3(1.f) - F) * T *
           fabs(distribution.D(wh) * distribution.G(wo, wi) * eta * eta *
                AbsDot(wi, wh) * fabs(dot(wo, wh)) * factor * factor /
                (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
}

template<typename Dist>
[[nodiscard]] Float MicrofacetTransmission_pdf(Dist distribution, Float etaA, Float etaB, Float3 wo, Float3 wi) noexcept {

    if (SameHemisphere(wo, wi)) return 0.0f;
    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    auto eta = CosTheta(wo) > 0.0f ? (etaB / etaA) : (etaA / etaB);
    auto wh = normalize(wo + wi * eta);

    // Compute change of variables _dwh\_dwi_ for microfacet transmission
    auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
    auto dwh_dwi = fabs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
    return distribution.Pdf(wo, wh) * dwh_dwi;
}

template<typename Dist>
[[nodiscard]] Float3 MicrofacetTransmission_sample(Dist distribution, Float3 T, Float etaA, Float etaB, Float3 wo, Float3* wi, Float2 u, Float* pdf) noexcept {

    if (wo.z == 0.0f) return make_Float3(0.0f);
    auto wh = distribution.Sample_wh(wo, u);
    if (dot(wo, wh) < 0.0f) return make_Float3(0.0f);  // Should be rare

    auto eta = CosTheta(wo) > 0.0f ? (etaA / etaB) : (etaB / etaA);
    if (!Refract(wo, wh, eta, wi)) return make_Float3(0.0f);
    *pdf = MicrofacetTransmission_pdf(distribution, etaA, etaB, wo, *wi);
    return MicrofacetTransmission_f(distribution, T, etaA, etaB, wo, *wi);
}

[[nodiscard]] Float3 SpecularTransmission_sample(Float3 T, Float etaA, Float etaB, Float3 wo, Float3* wi, Float2 sample, Float* pdf) noexcept {

    // Figure out which $\eta$ is incident and which is transmitted
    auto entering = CosTheta(wo) > 0.0f;
    auto etaI = entering ? etaA : etaB;
    auto etaT = entering ? etaB : etaA;

    // Compute ray direction for specular transmission
    if (!Refract(wo, Faceforward(make_Float3(0.0f, 0.0f, 1.0f), wo), etaI / etaT, wi)) {
        return make_Float3(0.0f);
    }
    *pdf = 1.0f;
    auto ft = T * (make_Float3(1.0f) - FresnelDielectric_f(etaA, etaB, CosTheta(*wi)));
    return ft / AbsCosTheta(*wi);
}

[[nodiscard]] Float3 LambertianTransmission_f(Float3 T) {
    return T * M_1_PIf;
}

[[nodiscard]] Float LambertianTransmission_pdf(Float3 wo, Float3 wi) {
    return !SameHemisphere(wo, wi) ? AbsCosTheta(wi) * M_1_PIf : 0.0f;
}

[[nodiscard]] Float3 LambertianTransmission_sample(Float3 T, Float3 wo, Float3 *wi, Float2 u, Float *pdf) {
    *wi = gr::cosineSampleHemisphere(u.x, u.y);
    if (wo.z > 0.0f) wi->z *= -1.0f;
    *pdf = LambertianTransmission_pdf(wo, *wi);
    return LambertianTransmission_f(T);
}

inline namespace materialflags {
constexpr auto DIFFUSE_BIT = 1u;
constexpr auto RETRO_BIT = 2u;
constexpr auto SHEEN_BIT = 4u;
constexpr auto CLEARCOAT_BIT = 8u;
constexpr auto SUBSURFACE_BIT = 16u;
constexpr auto MICROFACET_REFL_BIT = 32u;
constexpr auto MICROFACET_TRANS_BIT = 64u;
constexpr auto FAKE_SS_BIT = 128u;
constexpr auto LAMBERTIAN_TRANS_BIT = 256u;
}

class Material {

private:
    Onb _onb;
    uint32_t _flags;
    uint32_t _lobe_count;
    Diffuse _diffuse;
    FakeSS _fake_ss;
    Retro _retro;
    Sheen _sheen;
    Clearcoat _clearcoat;
    MicrofacetDistribution _distrib;
    MicrofacetDistribution _strans_distrib;
    Fresnel _fresnel;
    Float3 _trans;
    Float3 _lambert_trans;
    Float _e;

    [[nodiscard]] Float3 evaluate_local(Float3 wo, Float3 wi, uint32_t flags, Float *p_pdf, uint32_t *p_lobe_count) const noexcept {

        auto &lobe_count = *p_lobe_count;
        auto &pdf = *p_pdf;
        auto f = make_Float3(0.0f, 0.0f, 0.0f);
        pdf = 0.0f;
        lobe_count = 0u;
        if (wi.z * wo.z > 0.0f) {// reflect
            if (flags & DIFFUSE_BIT) {
                f += _diffuse.f(wo, wi);
                pdf += _diffuse.pdf(wo, wi);
                lobe_count++;
            }
            if (flags & RETRO_BIT) {
                f += _retro.f(wo, wi);
                pdf += _retro.pdf(wo, wi);
                lobe_count++;
            }
            if (flags & SHEEN_BIT) {
                f += _sheen.f(wo, wi);
                pdf += _sheen.pdf(wo, wi);
                lobe_count++;
            }
            if (flags & CLEARCOAT_BIT) {
                f += _clearcoat.f(wo, wi);
                pdf += _clearcoat.pdf(wo, wi);
                lobe_count++;
            }
            if (flags & MICROFACET_REFL_BIT) {
                f += MicrofacetReflection_f(_distrib, _fresnel, make_Float3(1.0f), wo, wi);
                pdf += MicrofacetReflection_pdf(_distrib, wo, wi);
                lobe_count++;
            }
            if (flags & FAKE_SS_BIT) {
                f += _fake_ss.f(wo, wi);
                pdf += _fake_ss.pdf(wo, wi);
                lobe_count++;
            }
        } else {
            if (flags & MICROFACET_TRANS_BIT) {
                f += MicrofacetTransmission_f(_strans_distrib, _trans, 1.0f, _e, wo, wi);
                pdf += MicrofacetTransmission_pdf(_strans_distrib, 1.0f, _e, wo, wi);
                lobe_count++;
            }
            if (flags & LAMBERTIAN_TRANS_BIT) {
                f += LambertianTransmission_f(_lambert_trans);
                pdf += LambertianTransmission_pdf(wo, wi);
                lobe_count++;
            }
        }
        return f;
    }

public:
    [[nodiscard]] Material(const gr::MaterialParameter &mtl, const gr::Intersection &isect) noexcept {

        auto two_sided = mtl.two_sided;
        auto front_face = two_sided || isect.faceSign > 0.0f;

        auto n = isect.ns;
        if (isect.has_differentials && false) {
            auto local_y = normalize(cross(n, normalize(isect.dpdu)));
            auto local_x = normalize(cross(local_y, n));

            if (isnan(local_x.x) || isnan(local_x.y) || isnan(local_x.z)) {
                printf("nan local_x %f %f %f\n", local_x.x, local_x.y, local_x.z);
                printf("nan dpdu %f %f %f\n", isect.dpdu.x, isect.dpdu.y, isect.dpdu.z);
            }
            if (isnan(local_y.x) || isnan(local_y.y) || isnan(local_y.z)) {
                printf("nan local_y %f %f %f\n", local_y.x, local_y.y, local_y.z);
                printf("nan dpdu %f %f %f\n", isect.dpdu.x, isect.dpdu.y, isect.dpdu.z);
            }
            _onb = Onb{local_x, local_y, n};
        } else {
            _onb = Onb{n};
        }

        _flags = 0u;
        _lobe_count = 0u;

        // Diffuse
        auto c = make_Float3(mtl.base_color);
        auto metallicWeight = mtl.metallic;
        _e = mtl.ior;
        auto strans = mtl.transmittance;
        auto diffuseWeight = (1.0f - metallicWeight) * (1.0f - strans);
        auto rough = mtl.roughness;
        auto lum = luminance(c);
        auto Ctint = lum > 0.0f ? (c / lum) : make_Float3(1.0f);
        auto dt = mtl.diffuse_trans;

        auto sheenWeight = mtl.sheen;
        auto Csheen = make_Float3(0.0f);
        if (front_face && sheenWeight > 0.0f) {
            auto stint = mtl.sheen_tint;
            Csheen = helpers::Lerp(stint, make_Float3(1.0f), Ctint);
        }

        // Create the microfacet distribution for metallic and/or specular
        // transmission.
        auto anisotropic = isect.has_differentials ? mtl.anisotropic : 0.0f;
        auto aspect = sqrtf(1.0f - anisotropic * 0.9f);
        auto ax = fmaxf(.001f, helpers::sqr(rough) / aspect);
        auto ay = fmaxf(.001f, helpers::sqr(rough) * aspect);
        _distrib = MicrofacetDistribution{ax, ay};

        // reflect
        if (front_face && diffuseWeight > 0.0f) {

            if (two_sided && mtl.thin) {
                auto flat = mtl.flatness;
                _diffuse = Diffuse{diffuseWeight * (1 - flat) * (1 - dt) * c};
                _flags |= DIFFUSE_BIT;
                _lobe_count++;
                _fake_ss = FakeSS{diffuseWeight * flat * (1 - dt) * c, rough};
                _flags |= FAKE_SS_BIT;
                _lobe_count++;
            } else {
                auto sd = mtl.scatter_distance;
                if (mtl.two_sided || (sd.x == 0.0f && sd.y == 0.0f && sd.z == 0.0f)) {// no BSSRDF
                    _diffuse = Diffuse{diffuseWeight * c};
                    _flags |= DIFFUSE_BIT;
                    _lobe_count++;
                } else {// subsurface
                    _flags |= SUBSURFACE_BIT;
                    _lobe_count++;
                }
            }

            // Retro-reflection.
            _retro = Retro{diffuseWeight * c, rough};
            _flags |= RETRO_BIT;
            _lobe_count++;

            if (sheenWeight > 0.0f) {
                _sheen = Sheen{diffuseWeight * sheenWeight * Csheen};
                _flags |= SHEEN_BIT;
                _lobe_count++;
            }
        }

        // Clearcoat
        auto cc = mtl.clearcoat;
        if (front_face && cc > 0.0f) {
            _flags |= CLEARCOAT_BIT;
            _lobe_count++;
            _clearcoat = Clearcoat{cc, Lerp(mtl.clearcoat_gloss, 0.1f, 0.001f)};
        }

        // Specular is Trowbridge-Reitz with a modified Fresnel function.
        auto specTint = mtl.specular_tint;
        auto Cspec0 = Lerp(metallicWeight, SchlickR0FromEta(_e) * Lerp(specTint, make_Float3(1.0f), Ctint), c);
        _fresnel = Fresnel{Cspec0, metallicWeight, _e};
        _flags |= MICROFACET_REFL_BIT;
        _lobe_count++;

        // trans
        if (strans > 0.0f) {
            _trans = strans * make_Float3(sqrtf(c.x), sqrtf(c.y), sqrtf(c.z));
            if (two_sided && mtl.thin) {
                // Scale roughness based on IOR (Burley 2015, Figure 15).
                auto rscaled = (0.65f * _e - 0.35f) * rough;
                auto ax = fmaxf(.001f, helpers::sqr(rscaled) / aspect);
                auto ay = fmaxf(.001f, helpers::sqr(rscaled) * aspect);
                _strans_distrib = MicrofacetDistribution{ax, ay};
                _flags |= MICROFACET_TRANS_BIT;
                _lobe_count++;
            } else if (!two_sided) {
                _strans_distrib = _distrib;
                _flags |= MICROFACET_TRANS_BIT;
                _lobe_count++;
            }
        }

        if (two_sided && mtl.thin && dt > 0.0f) {
            _lambert_trans = (1.0f - strans) * dt * c;
            _flags |= LAMBERTIAN_TRANS_BIT;
            _lobe_count++;
        }
    }

    [[nodiscard]] Float3 evaluate(Float3 wo_world, Float3 wi_world, Float *pdf) const noexcept {
        auto wo = _onb.transform(wo_world);
        auto wi = _onb.transform(wi_world);
        auto lobe_count = 0u;
        auto f = evaluate_local(wo, wi, _flags, pdf, &lobe_count);
        *pdf /= max(lobe_count, 1u);
        return f * helpers::AbsCosTheta(wi);
    }

    [[nodiscard]] Float3 sample(Float3 wo_world, Float3 *wi_world, Float2 u, Float *pdf, bool *is_delta, bool *is_bssrdf) const noexcept {

        auto wo = _onb.transform(wo_world);
        auto selected_lobe = clamp(static_cast<uint32_t>(u.x * _lobe_count), 0u, _lobe_count - 1u);
        auto selected_pdf = 0.0f;
        auto wi = make_Float3(0.0f, 0.0f, 1.0f);
        auto remapped_ux = u.x * _lobe_count - selected_lobe;
        auto u2 = make_Float2(remapped_ux, u.y);

        auto selected_flag = 0u;
        auto selected_f = [&] {
            auto l = 0u;
            if ((_flags & DIFFUSE_BIT) && l++ == selected_lobe) {
                selected_flag = DIFFUSE_BIT;
                return _diffuse.sample_f(wo, &wi, u2, &selected_pdf);
            }
            if ((_flags & RETRO_BIT) && l++ == selected_lobe) {
                selected_flag = RETRO_BIT;
                return _retro.sample_f(wo, &wi, u2, &selected_pdf);
            }
            if ((_flags & SHEEN_BIT) && l++ == selected_lobe) {
                selected_flag = SHEEN_BIT;
                return _sheen.sample_f(wo, &wi, u2, &selected_pdf);
            }
            if ((_flags & CLEARCOAT_BIT) && l++ == selected_lobe) {
                selected_flag = CLEARCOAT_BIT;
                return _clearcoat.sample_f(wo, &wi, u2, &selected_pdf);
            }
            if ((_flags & SUBSURFACE_BIT) && l++ == selected_lobe) {
                selected_flag = SUBSURFACE_BIT;
                return SpecularTransmission_sample(make_Float3(1.0f), 1.0f, _e, wo, &wi, u2, &selected_pdf);
            }
            if ((_flags & MICROFACET_REFL_BIT) && l++ == selected_lobe) {
                selected_flag = MICROFACET_REFL_BIT;
                return MicrofacetReflection_sample(_distrib, _fresnel, make_Float3(1.0f), wo, &wi, u2, &selected_pdf);
            }
            if ((_flags & MICROFACET_TRANS_BIT) && l++ == selected_lobe) {
                selected_flag = MICROFACET_TRANS_BIT;
                return MicrofacetTransmission_sample(_strans_distrib, _trans, 1.0f, _e, wo, &wi, u2, &selected_pdf);
            }
            if ((_flags & FAKE_SS_BIT) && l++ == selected_lobe) {
                selected_flag = FAKE_SS_BIT;
                return _fake_ss.sample_f(wo, &wi, u2, &selected_pdf);
            }
            if ((_flags & LAMBERTIAN_TRANS_BIT) && l++ == selected_lobe) {
                selected_flag = LAMBERTIAN_TRANS_BIT;
                return LambertianTransmission_sample(_lambert_trans, wo, &wi, u2, &selected_pdf);
            }
            return make_Float3(0.0f);
        }();

        auto inv_selection_prob = static_cast<Float>(_lobe_count);
        if (selected_flag == SUBSURFACE_BIT) {// specular transmittance only...
            *pdf = selected_pdf;
            *wi_world = _onb.inverseTransform(wi);
            *is_delta = true;
            *is_bssrdf = true;
            return selected_f * AbsCosTheta(wi) * inv_selection_prob;// scale according to selection prob
        }

        auto other_flags = _flags & ~selected_flag;
        auto other_pdf = 0.0f;
        auto other_lobe_count = 0u;
        auto other_f = evaluate_local(wo, wi, other_flags, &other_pdf, &other_lobe_count);
        auto inv_n = 1.0f / (other_lobe_count + 1u);
        *pdf = (selected_pdf + other_pdf) * inv_n;
        *wi_world = _onb.inverseTransform(wi);
        *is_delta = false;
        *is_bssrdf = false;
        return (selected_f + other_f) * AbsCosTheta(wi) * inv_n * inv_selection_prob;
    }

};

extern "C" [[nodiscard]] Float __direct_callable__pdf(const gr::MaterialParameter & mtl, const gr::Intersection & isect, Float3 wo_world, Float3 wi_world) {
    if (mtl.pass == gr::MaterialSamplePass::BSDF_PASS) {
        disney::Material material{ mtl, isect };
        auto pdf = 0.0f;
        [[maybe_unused]] auto f = material.evaluate(wo_world, wi_world, &pdf);
        return pdf;
    }
    else {
        return clamp(dot(wi_world, isect.ns), 0.0f, 1.0f);
    }
}

extern "C" [[nodiscard]] gr::MaterialEvaluation __direct_callable__eval(const gr::MaterialParameter & mtl, const gr::Intersection & isect, Float3 wo_world, Float3 wi_world) {

    if (mtl.pass == gr::MaterialSamplePass::BSDF_PASS) {  // BSDF direct lighting
        disney::Material material{ mtl, isect };
        auto pdf = 0.0f;
        auto f = material.evaluate(wo_world, wi_world, &pdf);
        return {f, pdf};
    }
    else {  // BSSRDF direct lighting, eval Sw
        auto cos_theta = dot(wi_world, isect.ns);
        if (cos_theta < 1e-4f) { return { make_Float3(0.0f), 0.0f }; }
        auto f = disney::SeparableBSSRDF_Sw(cos_theta, mtl.ior) * cos_theta;
        auto pdf = cos_theta * M_1_PIf;
        return { f, pdf };
    }
}

extern "C" [[nodiscard]] gr::MaterialSample __direct_callable__sample(OptixTraversableHandle scene, const gr::MaterialParameter & mtl, const gr::Intersection & isect, Float3 wo_world, Float time, Float2 u) {

    gr::MaterialSample s{};
    s.is_specular = false;

    if (mtl.pass == gr::MaterialSamplePass::BSDF_PASS) {
        disney::Material material{ mtl, isect };
        s.pi = isect.p;
        s.f = material.sample(wo_world, &s.wi, u, &s.pdf, &s.is_specular, &s.has_bssrdf);
        s.is_trans = dot(s.wi, isect.ns) * dot(wo_world, isect.ns) < 0.0f;
    }
    else if (mtl.pass == gr::MaterialSamplePass::BSSRDF_PASS) {  // handle BSSRDF here
        auto diffuseWeight = (1 - mtl.metallic) * (1 - mtl.transmittance);
        auto sd = mtl.scatter_distance;
        if (diffuseWeight > 0.0f && (sd.x != 0.0f || sd.y != 0.0f || sd.z != 0.0f)) {  // has subsurface
            sd.x = fmaxf(sd.x, 1e-3f);
            sd.y = fmaxf(sd.y, 1e-3f);
            sd.z = fmaxf(sd.z, 1e-3f);
            disney::BSSRDF bssrdf{ isect, make_Float3(mtl.base_color) * diffuseWeight, sd, mtl.ior };
            return bssrdf.Sample_S(scene, time, mtl.bssrdf_samples, u);
        }
    } else {
        auto wi = gr::cosineSampleHemisphere(u.x, u.y);
        s.f = disney::SeparableBSSRDF_Sw(wi.z /* = CosTheta(wi) */, mtl.ior) * helpers::AbsCosTheta(wi);
        s.pdf = helpers::AbsCosTheta(wi) * M_1_PIf;
        s.wi = Onb{ isect.ns }.inverseTransform(wi);  // local to world
    }

    return s;
}

}
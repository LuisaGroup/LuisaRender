//
// Created by Mike Smith on 2022/1/31.
//

#pragma once

#include <dsl/syntax.h>
#include <core/stl.h>
#include <util/spec.h>

namespace luisa::render {

using compute::Bool;
using compute::Expr;
using compute::Float;
using compute::Float2;
using compute::Float3;
using compute::Float4;

[[nodiscard]] Float3 reflect(Float3 wo, Float3 n) noexcept;
[[nodiscard]] Bool refract(Float3 wi, Float3 n, Float eta, Float3 *wt) noexcept;
[[nodiscard]] Float3 face_forward(Float3 v, Float3 n) noexcept;

[[nodiscard]] Float3 spherical_direction(Float sinTheta, Float cosTheta, Float phi) noexcept;
[[nodiscard]] Float3 spherical_direction(Float sinTheta, Float cosTheta, Float phi, Float3 x, Float3 y, Float3 z) noexcept;
[[nodiscard]] Float spherical_theta(Float3 v) noexcept;
[[nodiscard]] Float spherical_phi(Float3 v) noexcept;

class MicrofacetDistribution {

public:
    struct Gradient {
        Float2 dAlpha;
    };

private:
    Float2 _alpha;

public:
    explicit MicrofacetDistribution(Expr<float2> alpha) noexcept;
    virtual ~MicrofacetDistribution() noexcept = default;
    [[nodiscard]] Float G1(Expr<float3> w) const noexcept;
    [[nodiscard]] virtual Float G(Expr<float3> wo, Expr<float3> wi) const noexcept;
    [[nodiscard]] virtual Float D(Expr<float3> wh) const noexcept = 0;
    [[nodiscard]] virtual Float Lambda(Expr<float3> w) const noexcept = 0;
    [[nodiscard]] virtual Float3 sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept = 0;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wh) const noexcept;
    [[nodiscard]] auto alpha() const noexcept { return _alpha; }

    [[nodiscard]] virtual Gradient grad_G1(Expr<float3> w) const noexcept;
    [[nodiscard]] virtual Gradient grad_G(Expr<float3> wo, Expr<float3> wi) const noexcept;
    [[nodiscard]] virtual Gradient grad_D(Expr<float3> wh) const noexcept = 0;
    [[nodiscard]] virtual Gradient grad_Lambda(Expr<float3> w) const noexcept = 0;
};

struct TrowbridgeReitzDistribution : public MicrofacetDistribution {
    explicit TrowbridgeReitzDistribution(Expr<float2> alpha) noexcept;
    [[nodiscard]] Float D(Expr<float3> wh) const noexcept override;
    [[nodiscard]] Float Lambda(Expr<float3> w) const noexcept override;
    [[nodiscard]] Float3 sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept override;
    [[nodiscard]] static Float roughness_to_alpha(Expr<float> roughness) noexcept;
    [[nodiscard]] static Float2 roughness_to_alpha(Expr<float2> roughness) noexcept;

    [[nodiscard]] Gradient grad_D(Expr<float3> wh) const noexcept override;
    [[nodiscard]] Gradient grad_Lambda(Expr<float3> w) const noexcept override;
    [[nodiscard]] static Float2 grad_alpha_roughness(Expr<float2> roughness) noexcept;
};

[[nodiscard]] Float fresnel_dielectric(Float cosThetaI, Float etaI, Float etaT) noexcept;
[[nodiscard]] Float fresnel_conductor(Float cosThetaI, Float etaI, Float etaT, Float k) noexcept;
[[nodiscard]] SampledSpectrum fresnel_dielectric(
    Float cosThetaI, const SampledSpectrum &etaI, const SampledSpectrum &etaT) noexcept;
[[nodiscard]] SampledSpectrum fresnel_conductor(
    Float cosThetaI, const SampledSpectrum &etaI,
    const SampledSpectrum &etaT, const SampledSpectrum &k) noexcept;

struct Fresnel {
    struct Gradient {
        virtual ~Gradient() noexcept = default;
    };

    virtual ~Fresnel() noexcept = default;
    [[nodiscard]] virtual SampledSpectrum evaluate(Expr<float> cosI) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Gradient> backward(Expr<float> cosI, const SampledSpectrum &df) const noexcept {
        return luisa::make_unique<Fresnel::Gradient>();
    }
};

class FresnelConductor final : public Fresnel {

private:
    SampledSpectrum _eta_i;
    SampledSpectrum _eta_t;
    SampledSpectrum _k;

public:
    FresnelConductor(SampledSpectrum etaI, SampledSpectrum etaT, SampledSpectrum k) noexcept
        : _eta_i{std::move(etaI)}, _eta_t{std::move(etaT)}, _k{std::move(k)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosThetaI) const noexcept override;
};

class FresnelDielectric final : public Fresnel {

private:
    SampledSpectrum _eta_i, _eta_t;

public:
    FresnelDielectric(SampledSpectrum etaI, SampledSpectrum etaT) noexcept
        : _eta_i{std::move(etaI)}, _eta_t{std::move(etaT)} {}
    [[nodiscard]] auto eta_i() const noexcept { return _eta_i; }
    [[nodiscard]] auto eta_t() const noexcept { return _eta_t; }
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosThetaI) const noexcept override;
};

struct BxDF {
    virtual ~BxDF() noexcept = default;
    [[nodiscard]] virtual SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept = 0;
    [[nodiscard]] virtual SampledSpectrum sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept;
    [[nodiscard]] virtual Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept;
};

class LambertianReflection : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dR;
    };

private:
    SampledSpectrum _r;

public:
    explicit LambertianReflection(SampledSpectrum R) noexcept : _r{std::move(R)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
};

class LambertianTransmission : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dT;
    };

private:
    SampledSpectrum _t;

public:
    explicit LambertianTransmission(SampledSpectrum T) noexcept : _t{std::move(T)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] SampledSpectrum sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] static Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) noexcept;
};

class MicrofacetReflection : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dR;
        Float2 dAlpha;

        luisa::unique_ptr<Fresnel::Gradient> dFresnel;
    };

private:
    // MicrofacetReflection Private Data
    SampledSpectrum _r;
    const MicrofacetDistribution *_distribution;
    const Fresnel *_fresnel;

public:
    MicrofacetReflection(SampledSpectrum R, const MicrofacetDistribution *d, const Fresnel *f) noexcept
        : _r{std::move(R)}, _distribution{d}, _fresnel{f} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] SampledSpectrum sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
};

class MicrofacetTransmission : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dT;
        Float2 dAlpha;
    };

private:
    // MicrofacetTransmission Private Data
    SampledSpectrum _t;
    const MicrofacetDistribution *_distribution;
    SampledSpectrum _eta_a;
    SampledSpectrum _eta_b;

public:
    // MicrofacetTransmission Public Methods
    MicrofacetTransmission(SampledSpectrum T, const MicrofacetDistribution *d,
                           SampledSpectrum etaA, SampledSpectrum etaB) noexcept
        : _t{std::move(T)}, _distribution{d}, _eta_a{std::move(etaA)}, _eta_b{std::move(etaB)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] SampledSpectrum sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
};

class OrenNayar : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dR;
        Float dSigma;
    };

private:
    SampledSpectrum _r;
    Float _sigma;
    Float _a;
    Float _b;

public:
    OrenNayar(SampledSpectrum R, Expr<float> sigma) noexcept;
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
};

class FresnelBlend : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dRd;
        SampledSpectrum dRs;
        Float2 dAlpha;
    };

private:
    // FresnelBlend Private Data
    SampledSpectrum _rd;
    SampledSpectrum _rs;
    Float _rd_ratio;
    const MicrofacetDistribution *_distribution;

private:
    [[nodiscard]] SampledSpectrum Schlick(Expr<float> cosTheta) const noexcept;

public:
    FresnelBlend(SampledSpectrum Rd, SampledSpectrum Rs, const MicrofacetDistribution *distrib, Expr<float> Rd_sample_ratio = .5f) noexcept
        : _rd{std::move(Rd)}, _rs{std::move(Rs)}, _rd_ratio{clamp(Rd_sample_ratio, .05f, .95f)},  _distribution{distrib} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] SampledSpectrum sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
};

}// namespace luisa::render

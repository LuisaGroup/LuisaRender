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

[[nodiscard]] Bool refract(Float3 wi, Float3 n, Float eta, Float3 *wt) noexcept;
[[nodiscard]] Float3 face_forward(Float3 v, Float3 n) noexcept;

[[nodiscard]] Float3 spherical_direction(Float sinTheta, Float cosTheta, Float phi) noexcept;
[[nodiscard]] Float3 spherical_direction(Float sinTheta, Float cosTheta, Float phi, Float3 x, Float3 y, Float3 z) noexcept;
[[nodiscard]] Float spherical_theta(Float3 v) noexcept;
[[nodiscard]] Float spherical_phi(Float3 v) noexcept;

enum TransportMode {
    RADIANCE,
    IMPORTANCE
};

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
    [[nodiscard]] static Float alpha_to_roughness(Expr<float> alpha) noexcept;
    [[nodiscard]] static Float2 alpha_to_roughness(Expr<float2> alpha) noexcept;

    [[nodiscard]] Gradient grad_D(Expr<float3> wh) const noexcept override;
    [[nodiscard]] Gradient grad_Lambda(Expr<float3> w) const noexcept override;
    [[nodiscard]] static Float2 grad_alpha_roughness(Expr<float2> roughness) noexcept;
};

[[nodiscard]] Float fresnel_dielectric(Float cosThetaI, Float etaI, Float etaT) noexcept;
[[nodiscard]] SampledSpectrum fresnel_conductor(Float cosThetaI, Float etaI,
                                                const SampledSpectrum &etaT,
                                                const SampledSpectrum &k) noexcept;

// approx. integration of Fr(cosTheta) * cosTheta
[[nodiscard]] Float fresnel_dielectric_integral(Float eta) noexcept;

struct Fresnel {
    struct Gradient {
        virtual ~Gradient() noexcept = default;
    };

    virtual ~Fresnel() noexcept = default;
    [[nodiscard]] virtual SampledSpectrum evaluate(Expr<float> cosI) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Gradient> backward(
        Expr<float> cosI, const SampledSpectrum &df) const noexcept {
        return nullptr;
    }
};

class FresnelConductor final : public Fresnel {

private:
    Float _eta_i;
    SampledSpectrum _eta_t;
    SampledSpectrum _k;

public:
    FresnelConductor(Expr<float> etaI, const SampledSpectrum &etaT, const SampledSpectrum &k) noexcept
        : _eta_i{etaI}, _eta_t{etaT}, _k{k} {}
    [[nodiscard]] auto eta_i() const noexcept { return _eta_i; }
    [[nodiscard]] auto eta_t() const noexcept { return _eta_t; }
    [[nodiscard]] auto k() const noexcept { return _k; }
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosThetaI) const noexcept override;
};

class FresnelDielectric final : public Fresnel {

private:
    Float _eta_i, _eta_t;

public:
    FresnelDielectric(Expr<float> etaI, Expr<float> etaT) noexcept
        : _eta_i{etaI}, _eta_t{etaT} {}
    [[nodiscard]] auto eta_i() const noexcept { return _eta_i; }
    [[nodiscard]] auto eta_t() const noexcept { return _eta_t; }
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosThetaI) const noexcept override;
};

class BxDF {

public:
    struct SampledDirection {
        Float3 wi;
        Bool valid;
    };

public:
    virtual ~BxDF() noexcept = default;
    [[nodiscard]] virtual SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept = 0;
    [[nodiscard]] virtual SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept;
    [[nodiscard]] virtual SampledSpectrum sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf, TransportMode mode) const noexcept;
    [[nodiscard]] virtual Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept;
    [[nodiscard]] virtual SampledSpectrum albedo() const noexcept = 0;
};

class LambertianReflection : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dR;
    };

private:
    SampledSpectrum _r;

public:
    explicit LambertianReflection(const SampledSpectrum &R) noexcept : _r{R} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _r; }
};

class LambertianTransmission : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dT;
    };

private:
    SampledSpectrum _t;

public:
    explicit LambertianTransmission(const SampledSpectrum &T) noexcept : _t{T} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] static Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) noexcept;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return SampledSpectrum(0.f); }
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
    MicrofacetReflection(const SampledSpectrum &R, const MicrofacetDistribution *d, const Fresnel *f) noexcept
        : _r{R}, _distribution{d}, _fresnel{f} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _r; }
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
    Float _eta_a;
    Float _eta_b;

public:
    // MicrofacetTransmission Public Methods
    MicrofacetTransmission(const SampledSpectrum &T,
                           const MicrofacetDistribution *d,
                           Expr<float> etaA, Expr<float> etaB) noexcept
        : _t{T}, _distribution{d}, _eta_a{etaA}, _eta_b{etaB} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode) const noexcept;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return SampledSpectrum(0.f); }
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
    OrenNayar(const SampledSpectrum &R, Expr<float> sigma) noexcept;
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _r; }
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
    FresnelBlend(const SampledSpectrum &Rd, const SampledSpectrum &Rs,
                 const MicrofacetDistribution *distrib, Expr<float> Rd_sample_ratio = .5f) noexcept
        : _rd{Rd}, _rs{Rs}, _rd_ratio{clamp(Rd_sample_ratio, .05f, .95f)}, _distribution{distrib} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u, TransportMode mode) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override;
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept;
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _rd; }
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::BxDF)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::BxDF::SampledDirection)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::MicrofacetDistribution)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Fresnel)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::FresnelConductor)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::FresnelDielectric)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::LambertianReflection)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::LambertianTransmission)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::MicrofacetReflection)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::MicrofacetTransmission)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::OrenNayar)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::FresnelBlend)

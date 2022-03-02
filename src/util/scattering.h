//
// Created by Mike Smith on 2022/1/31.
//

#pragma once

#include <dsl/syntax.h>
#include <core/stl.h>

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

struct MicrofacetDistribution {
    virtual ~MicrofacetDistribution() noexcept = default;
    [[nodiscard]] Float G1(Expr<float3> w) const noexcept;
    [[nodiscard]] virtual Float G(Expr<float3> wo, Expr<float3> wi) const noexcept;
    [[nodiscard]] virtual Float D(Expr<float3> wh) const noexcept = 0;
    [[nodiscard]] virtual Float Lambda(Expr<float3> w) const noexcept = 0;
    [[nodiscard]] virtual Float3 sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept = 0;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wh) const noexcept;
};

class TrowbridgeReitzDistribution : public MicrofacetDistribution {

private:
    Float2 _alpha;

public:
    explicit TrowbridgeReitzDistribution(Expr<float2> alpha) noexcept;
    [[nodiscard]] Float D(Expr<float3> wh) const noexcept override;
    [[nodiscard]] Float Lambda(Expr<float3> w) const noexcept override;
    [[nodiscard]] Float3 sample_wh(Expr<float3> wo, Expr<float2> u) const noexcept override;
    [[nodiscard]] static Float roughness_to_alpha(Expr<float> roughness) noexcept;
    [[nodiscard]] static Float2 roughness_to_alpha(Expr<float2> roughness) noexcept;
    [[nodiscard]] auto alpha() const noexcept { return _alpha; }
};

[[nodiscard]] Float fresnel_dielectric(Float cosThetaI, Float etaI, Float etaT) noexcept;
[[nodiscard]] Float4 fresnel_dielectric(Float cosThetaI, Float4 etaI, Float4 etaT) noexcept;
[[nodiscard]] Float4 fresnel_conductor(Float cosThetaI, Float4 etaI, Float4 etaT, Float4 k) noexcept;

struct Fresnel {
    virtual ~Fresnel() noexcept = default;
    [[nodiscard]] virtual Float4 evaluate(Expr<float> cosI) const noexcept = 0;
};

class FresnelConductor final : public Fresnel {

private:
    Float4 _eta_i, _eta_t, _k;

public:
    FresnelConductor(Expr<float4> etaI, Expr<float4> etaT, Expr<float4> k) noexcept
        : _eta_i{etaI}, _eta_t{etaT}, _k{k} {}
    [[nodiscard]] Float4 evaluate(Expr<float> cosThetaI) const noexcept override;
};

class FresnelDielectric final : public Fresnel {

private:
    Float4 _eta_i, _eta_t;

public:
    FresnelDielectric(Expr<float4> etaI, Expr<float4> etaT) noexcept
        : _eta_i{etaI}, _eta_t{etaT} {}
    [[nodiscard]] Float4 evaluate(Expr<float> cosThetaI) const noexcept override;
    [[nodiscard]] auto eta_i() const noexcept { return _eta_i; }
    [[nodiscard]] auto eta_t() const noexcept { return _eta_t; }
};

struct FresnelNoOp final : public Fresnel {
    [[nodiscard]] Float4 evaluate(Expr<float>) const noexcept override;
};

struct BxDF {
    virtual ~BxDF() noexcept = default;
    [[nodiscard]] virtual Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept = 0;
    [[nodiscard]] virtual Float4 sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept;
    [[nodiscard]] virtual Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept;
    [[nodiscard]] virtual luisa::map<luisa::string, Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept = 0;
};

class LambertianReflection : public BxDF {

private:
    Float4 _r;

public:
    explicit LambertianReflection(Expr<float4> R) noexcept : _r{R} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] map<luisa::string, Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override;
};

class LambertianTransmission : public BxDF {

private:
    Float4 _t;

public:
    explicit LambertianTransmission(Expr<float4> T) noexcept : _t{T} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Float4 sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] map<luisa::string, Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override;
};

class MicrofacetReflection : public BxDF {

private:
    // MicrofacetReflection Private Data
    Float4 _r;
    const MicrofacetDistribution *_distribution;
    const Fresnel *_fresnel;

public:
    MicrofacetReflection(Expr<float4> R, const MicrofacetDistribution *d, const Fresnel *f) noexcept
        : _r{R}, _distribution{d}, _fresnel{f} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Float4 sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] map<luisa::string, Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override;
};

class MicrofacetTransmission : public BxDF {

private:
    // MicrofacetTransmission Private Data
    Float4 _t;
    const MicrofacetDistribution *_distribution;
    Float4 _eta_a;
    Float4 _eta_b;
    FresnelDielectric _fresnel;

public:
    // MicrofacetTransmission Public Methods
    MicrofacetTransmission(
        Expr<float4> T, const MicrofacetDistribution *d,
        Expr<float4> etaA, Expr<float4> etaB) noexcept
        : _t{T}, _distribution{d}, _eta_a{etaA},
          _eta_b{etaB}, _fresnel{etaA, etaB} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Float4 sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] map<luisa::string, Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override;
};

class OrenNayar : public BxDF {

private:
    Float4 _r;
    Float _sigma;
    Float _a;
    Float _b;

public:
    OrenNayar(Expr<float4> R, Expr<float> sigma) noexcept;
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] map<luisa::string, Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override;
};

class FresnelBlend : public BxDF {

private:
    // FresnelBlend Private Data
    Float4 _rd, _rs;
    const MicrofacetDistribution *_distribution;

private:
    [[nodiscard]] Float4 Schlick(Expr<float> cosTheta) const noexcept;

public:
    FresnelBlend(Expr<float4> Rd, Expr<float4> Rs, const MicrofacetDistribution *distrib) noexcept
        : _rd{Rd}, _rs{Rs}, _distribution{distrib} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] Float4 sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *pdf) const noexcept override;
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override;
    [[nodiscard]] map<luisa::string, Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override;
};

}// namespace luisa::render

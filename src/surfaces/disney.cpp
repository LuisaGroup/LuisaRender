//
// Created by Mike Smith on 2022/1/30.
//

#include <util/sampling.h>
#include <base/surface.h>
#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class DisneySurface final : public Surface {

private:
    const Texture *_color{};
    const Texture *_metallic{};
    const Texture *_eta{};
    const Texture *_roughness{};
    const Texture *_specular_tint{};
    const Texture *_anisotropic{};
    const Texture *_sheen{};
    const Texture *_sheen_tint{};
    const Texture *_clearcoat{};
    const Texture *_clearcoat_gloss{};
    const Texture *_specular_trans{};
    const Texture *_flatness{};
    const Texture *_diffuse_trans{};

public:
    DisneySurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc} {
        auto load_texture = [scene, desc](const Texture *&t, std::string_view name, Texture::Category category) noexcept {
            if (category == Texture::Category::COLOR) {
                t = scene->load_texture(desc->property_node_or_default(
                    name, SceneNodeDesc::shared_default_texture("ConstColor")));
                if (t->category() != category) [[unlikely]] {
                    LUISA_ERROR(
                        "Expected color texture for "
                        "property '{}' in DisneySurface. [{}]",
                        name, desc->source_location().string());
                }
            } else {
                t = scene->load_texture(desc->property_node_or_default(
                    name, SceneNodeDesc::shared_default_texture("ConstGeneric")));
                if (t->category() != category) [[unlikely]] {
                    LUISA_ERROR(
                        "Expected generic texture for "
                        "property '{}' in DisneySurface. [{}]",
                        name, desc->source_location().string());
                }
            }
        };
#define LUISA_RENDER_DISNEY_PARAM_LOAD(name, category) \
    load_texture(_##name, #name, Texture::Category::category);
        LUISA_RENDER_DISNEY_PARAM_LOAD(color, COLOR)
        LUISA_RENDER_DISNEY_PARAM_LOAD(metallic, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(eta, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(roughness, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(specular_tint, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(anisotropic, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(sheen, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(sheen_tint, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(clearcoat, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(clearcoat_gloss, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(specular_trans, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(flatness, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(diffuse_trans, GENERIC)
#undef LUISA_RENDER_DISNEY_PARAM_LOAD
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] uint encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint instance_id, const Shape *shape) const noexcept override {
        std::array textures{
            *pipeline.encode_texture(command_buffer, _color),
            *pipeline.encode_texture(command_buffer, _metallic),
            *pipeline.encode_texture(command_buffer, _eta),
            *pipeline.encode_texture(command_buffer, _roughness),
            *pipeline.encode_texture(command_buffer, _specular_tint),
            *pipeline.encode_texture(command_buffer, _anisotropic),
            *pipeline.encode_texture(command_buffer, _sheen),
            *pipeline.encode_texture(command_buffer, _sheen_tint),
            *pipeline.encode_texture(command_buffer, _clearcoat),
            *pipeline.encode_texture(command_buffer, _clearcoat_gloss),
            *pipeline.encode_texture(command_buffer, _specular_trans),
            *pipeline.encode_texture(command_buffer, _flatness),
            *pipeline.encode_texture(command_buffer, _diffuse_trans)};
        auto [buffer, buffer_id] = pipeline.arena_buffer<TextureHandle>(textures.size());
        command_buffer << buffer.copy_from(&textures) << compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

namespace disney_impl {

using namespace compute;

[[nodiscard]] inline Float sqr(Float x) noexcept { return x * x; }
[[nodiscard]] inline auto Lerp(const auto &t, const auto &v1, const auto &v2) noexcept { return lerp(v1, v2, t); }
[[nodiscard]] inline Float AbsDot(Float3 u, Float3 v) noexcept { return abs(dot(u, v)); }
[[nodiscard]] inline Float CosTheta(Float3 w) { return w.z; }
[[nodiscard]] inline Float Cos2Theta(Float3 w) { return w.z * w.z; }
[[nodiscard]] inline Float AbsCosTheta(Float3 w) { return abs(w.z); }
[[nodiscard]] inline Float Sin2Theta(Float3 w) { return max(0.0f, 1.0f - Cos2Theta(w)); }
[[nodiscard]] inline Float SinTheta(Float3 w) { return sqrt(Sin2Theta(w)); }
[[nodiscard]] inline Float TanTheta(Float3 w) { return SinTheta(w) / CosTheta(w); }
[[nodiscard]] inline Float Tan2Theta(Float3 w) { return Sin2Theta(w) / Cos2Theta(w); }
[[nodiscard]] inline Float CosPhi(Float3 w) {
    auto sinTheta = SinTheta(w);
    return ite(sinTheta == 0.0f, 1.0f, clamp(w.x / sinTheta, -1.0f, 1.0f));
}
[[nodiscard]] inline Float SinPhi(Float3 w) {
    auto sinTheta = SinTheta(w);
    return ite(sinTheta == 0.0f, 0.0f, clamp(w.y / sinTheta, -1.0f, 1.0f));
}
[[nodiscard]] inline Float Cos2Phi(Float3 w) { return CosPhi(w) * CosPhi(w); }
[[nodiscard]] inline Float Sin2Phi(Float3 w) { return SinPhi(w) * SinPhi(w); }
[[nodiscard]] inline Bool SameHemisphere(Float3 w, Float3 wp) noexcept { return w.z * wp.z > 0.0f; }
[[nodiscard]] inline Float3 Faceforward(Float3 v, Float3 v2) noexcept { return sign(dot(v, v2)) * v; }

[[nodiscard]] inline Float3 SphericalDirection(Float sinTheta, Float cosTheta, Float phi) noexcept {
    return make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

[[nodiscard]] inline Float3 Reflect(Float3 wo, Float3 n) noexcept {
    return -wo + 2.0f * dot(wo, n) * n;
}

[[nodiscard]] inline Bool Refract(Float3 wi, Float3 n, Float eta, Float3 *wt) noexcept {
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    auto cosThetaI = dot(n, wi);
    auto sin2ThetaI = max(0.0f, 1.0f - cosThetaI * cosThetaI);
    auto sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    auto is_refract = sin2ThetaT < 1.0f;
    $if(is_refract) {
        auto cosThetaT = sqrt(1.0f - sin2ThetaT);
        *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    };
    return is_refract;
}

[[nodiscard]] inline Float SchlickWeight(Float cosTheta) noexcept {// (1 - cos)^5
    auto m = clamp(1.0f - cosTheta, 0.0f, 1.0f);
    return (m * m) * (m * m) * m;
}

[[nodiscard]] inline Float FrSchlick(const auto &R0, Float cosTheta) noexcept {
    return Lerp(SchlickWeight(cosTheta), R0, 1.0f);
}

[[nodiscard]] inline Float FresnelMoment1(Float eta) noexcept {
    auto eta2 = eta * eta;
    auto eta3 = eta2 * eta;
    auto eta4 = eta3 * eta;
    auto eta5 = eta4 * eta;
    return ite(
        eta < 1.0f,
        0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945f * eta3 +
            2.49277f * eta4 - 0.68441f * eta5,
        -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
            1.27198f * eta4 + 0.12746f * eta5);
}

[[nodiscard]] inline Float FrDielectric(Float cosThetaI, Float etaI_in, Float etaT_in) noexcept {
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
    // Potentially swap indices of refraction
    auto entering = cosThetaI > 0.f;
    auto etaI = ite(entering, etaI_in, etaT_in);
    auto etaT = ite(entering, etaT_in, etaI_in);
    cosThetaI = abs(cosThetaI);

    // Compute _cosThetaT_ using Snell's law
    auto sinThetaI = sqrt(1.0f - cosThetaI * cosThetaI);
    auto sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    auto fr = def(1.0f);
    $if(sinThetaT < 1.0f) {
        auto cosThetaT = sqrt(max(0.0f, 1.0f - sinThetaT * sinThetaT));
        auto Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                     ((etaT * cosThetaI) + (etaI * cosThetaT));
        auto Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                     ((etaI * cosThetaI) + (etaT * cosThetaT));
        fr = (Rparl * Rparl + Rperp * Rperp) * 0.5f;
    };
    return fr;
}

// For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
// coming from air..
[[nodiscard]] inline Float SchlickR0FromEta(Float eta) noexcept { return sqr(eta - 1.0f) / sqr(eta + 1.0f); }

[[nodiscard]] inline Float GTR1(Float cosTheta, Float alpha) noexcept {
    auto alpha2 = alpha * alpha;
    return (alpha2 - 1.0f) / (pi * log(alpha2) * (1.0f + (alpha2 - 1.0f) * cosTheta * cosTheta));
}

// Smith masking/shadowing term.
[[nodiscard]] inline Float smithG_GGX(Float cosTheta, Float alpha) noexcept {
    auto alpha2 = alpha * alpha;
    auto cosTheta2 = cosTheta * cosTheta;
    return 1.0f / (cosTheta + sqrt(alpha2 + cosTheta2 - alpha2 * cosTheta2));
}

struct Lobe {
    virtual ~Lobe() noexcept = default;
    [[nodiscard]] virtual Float4 f(Float3 wo, Float3 wi) const noexcept = 0;
    [[nodiscard]] virtual Float pdf(Float3 wo, Float3 wi) const noexcept {
        return ite(SameHemisphere(wo, wi), AbsCosTheta(wi) * inv_pi, 0.0f);
    }
    [[nodiscard]] virtual Float4 sample(Float3 wo, Float3 *wi, Float2 u, Float *pdf) const noexcept {
        auto w = sample_cosine_hemisphere(u);
        *wi = w;
        *pdf = w.z * inv_pi;
        return f(wo, w);
    }
};

class Diffuse final : public Lobe {

private:
    Float4 R;

public:
    explicit Diffuse(Float4 R = {}) noexcept : R{R} {}
    [[nodiscard]] Float4 f(Float3 wo, Float3 wi) const noexcept override {
        auto Fo = SchlickWeight(AbsCosTheta(wo));
        auto Fi = SchlickWeight(AbsCosTheta(wi));

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
        // Burley 2015, eq (4).
        return R * inv_pi * (1.0f - Fo * 0.5f) * (1.0f - Fi * 0.5f);
    }
};

class FakeSS final : public Lobe {

private:
    Float4 R;
    Float roughness;

public:
    explicit FakeSS(Float4 R = {}, Float roughness = {}) noexcept : R{R}, roughness{roughness} {}
    [[nodiscard]] Float4 f(Float3 wo, Float3 wi) const noexcept override {
        auto wh = wi + wo;
        auto F = def<float4>();
        $if(any(wh != 0.0f)) {
            wh = normalize(wh);
            auto cosThetaD = dot(wi, wh);
            // Fss90 used to "flatten" retro-reflection based on roughness
            auto Fss90 = cosThetaD * cosThetaD * roughness;
            auto Fo = SchlickWeight(AbsCosTheta(wo));
            auto Fi = SchlickWeight(AbsCosTheta(wi));
            auto Fss = Lerp(Fo, 1.0f, Fss90) * Lerp(Fi, 1.0f, Fss90);
            // 1.25 scale is used to (roughly) preserve albedo
            auto ss = 1.25f * (Fss * (1.0f / (AbsCosTheta(wo) + AbsCosTheta(wi)) - 0.5f) + 0.5f);
            F = R * inv_pi * ss;
        };
        return F;
    }
};

class Retro final : public Lobe {

private:
    Float4 R;
    Float roughness;

public:
    explicit Retro(Float4 R = {}, Float roughness = {}) noexcept : R{R}, roughness{roughness} {}
    [[nodiscard]] Float4 f(Float3 wo, Float3 wi) const noexcept override {
        auto wh = wi + wo;
        auto F = def<float4>();
        $if(any(wh != 0.0f)) {
            wh = normalize(wh);
            auto cosThetaD = dot(wi, wh);

            Float Fo = SchlickWeight(AbsCosTheta(wo)),
                  Fi = SchlickWeight(AbsCosTheta(wi));
            Float Rr = 2 * roughness * cosThetaD * cosThetaD;

            // Burley 2015, eq (4).
            F = R * inv_pi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1.0f));
        };
        return F;
    }
};

class Sheen final : public Lobe {

private:
    Float4 R;

public:
    [[nodiscard]] explicit Sheen(Float4 R = {}) noexcept : R{R} {}
    [[nodiscard]] Float4 f(Float3 wo, Float3 wi) const noexcept override {
        auto wh = wi + wo;
        auto F = def<float4>();
        $if(any(wh != 0.0f)) {
            wh = normalize(wh);
            auto cosThetaD = dot(wi, wh);
            F = R * SchlickWeight(cosThetaD);
        };
        return F;
    }
};

class Clearcoat final : public Lobe {

private:
    Float weight;
    Float gloss;

public:
    explicit Clearcoat(Float weight = {}, Float gloss = {}) noexcept
        : weight{weight}, gloss{gloss} {}
    [[nodiscard]] Float4 f(Float3 wo, Float3 wi) const noexcept override {
        auto wh = wi + wo;
        auto F = def<float4>();
        $if(any(wh != 0.0f)) {
            wh = normalize(wh);

            // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
            // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
            // (which is GTR2).
            auto Dr = GTR1(AbsCosTheta(wh), gloss);
            auto Fr = FrSchlick(.04f, dot(wo, wh));
            // The geometric term always based on alpha = 0.25.
            auto Gr = smithG_GGX(AbsCosTheta(wo), .25f) *
                      smithG_GGX(AbsCosTheta(wi), .25f);
            F = make_float4(weight * Gr * Fr * Dr * .25f);
        };
        return F;
    }
    [[nodiscard]] Float pdf(Float3 wo, Float3 wi) const noexcept override {
        auto wh = wi + wo;
        auto p = def(0.0f);
        $if(SameHemisphere(wo, wi) & any(wh != 0.0f)) {
            wh = normalize(wh);
            // The sampling routine samples wh exactly from the GTR1 distribution.
            // Thus, the final value of the PDF is just the value of the
            // distribution for wh converted to a measure with respect to the
            // surface normal.
            auto Dr = GTR1(AbsCosTheta(wh), gloss);
            p = Dr * AbsCosTheta(wh) / (4.0f * dot(wo, wh));
        };
        return p;
    }
    [[nodiscard]] Float4 sample(Float3 wo, Float3 *wi, Float2 u, Float *p) const noexcept override {
        // TODO: double check all this: there still seem to be some very
        // occasional fireflies with clearcoat; presumably there is a bug
        // somewhere.
        *p = 0.0f;
        auto F = def<float4>();
        $if(wo.z != 0.0f) {
            auto alpha2 = gloss * gloss;
            auto cosTheta = sqrt(max(0.0f, (1.0f - pow(alpha2, 1.0f - u.x)) / (1.0f - alpha2)));
            auto sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
            auto phi = 2 * pi * u.y;
            auto wh = SphericalDirection(sinTheta, cosTheta, phi);
            wh = normalize(ite(SameHemisphere(wo, wh), wh, -wh));
            auto w = Reflect(wo, wh);
            $if(SameHemisphere(wo, w)) {
                *wi = w;
                *p = pdf(wo, w);
                F = f(wo, w);
            };
        };
        return F;
    }
};

class Fresnel {

private:
    Float4 R0;
    Float metallic;
    Float eta;

public:
    [[nodiscard]] explicit Fresnel(Float4 R0 = {}, Float metallic = {}, Float eta = {}) noexcept
        : R0{R0}, metallic{metallic}, eta{eta} {}
    [[nodiscard]] Float4 evaluate(Float cosI) const noexcept {
        return Lerp(metallic, make_float4(FrDielectric(cosI, 1.0f, eta)), FrSchlick(R0, cosI));
    }
};

[[nodiscard]] inline Float2 TrowbridgeReitzSample11(Float cosTheta, Float U1, Float U2) noexcept {
    // special case (normal incidence)
    auto sample = def<float2>();
    $if(cosTheta > 0.9999f) {
        auto r = sqrt(U1 / (1.0f - U1));
        auto phi = 2.0f * pi * U2;
        auto slope_x = r * cos(phi);
        auto slope_y = r * sin(phi);
        sample = make_float2(slope_x, slope_y);
    }
    $else {
        auto sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
        auto tanTheta = sinTheta / cosTheta;
        auto a = 1.0f / tanTheta;
        auto G1 = 2.0f / (1.0f + sqrt(1.f + 1.f / (a * a)));
        // sample slope_x
        auto A = 2.0f * U1 / G1 - 1.0f;
        auto tmp = min(1.f / (A * A - 1.f), 1e10f);
        auto B = tanTheta;
        auto D = sqrt(max(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.0f));
        auto slope_x_1 = B * tmp - D;
        auto slope_x_2 = B * tmp + D;
        auto slope_x = ite(
            A<0.0f | slope_x_2> 1.f / tanTheta,
            slope_x_1, slope_x_2);
        // sample slope_y
        auto S = ite(U2 > 0.5f, 1.0f, -1.0f);
        U2 = ite(U2 > 0.5f, 2.f * (U2 - .5f), 2.f * (.5f - U2));
        auto z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
                 (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
        auto slope_y = S * z * sqrt(1.f + slope_x * slope_x);
        sample = make_float2(slope_x, slope_y);
    };
    return sample;
}

[[nodiscard]] inline Float3 TrowbridgeReitzSample(Float3 wi, Float alpha_x, Float alpha_y, Float U1, Float U2) {
    // 1. stretch wi
    auto wiStretched = normalize(make_float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));
    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    auto slope = TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2);
    auto slope_x = slope.x;
    auto slope_y = slope.y;
    // 3. rotate
    auto tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;
    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;
    // 5. compute normal
    return normalize(make_float3(-slope_x, -slope_y, 1.0f));
}

struct MicrofacetDistribution {
    virtual ~MicrofacetDistribution() noexcept = default;
    [[nodiscard]] virtual Float G(Float3 wo, Float3 wi) const noexcept = 0;
    [[nodiscard]] virtual Float D(Float3 wh) const noexcept = 0;
    [[nodiscard]] virtual Float pdf(Float3 wo, Float3 wh) const noexcept = 0;
    [[nodiscard]] virtual Float3 sample_wh(Float3 wo, Float2 u) const noexcept = 0;
};

class TrowbridgeReitzDistribution : public MicrofacetDistribution {

protected:
    Float alphax, alphay;

protected:
    [[nodiscard]] Float G1(Float3 w) const noexcept {
        return 1.0f / (1.0f + Lambda(w));
    }

private:
    [[nodiscard]] Float Lambda(const Float3 &w) const noexcept {
        auto L = def(0.0f);
        auto absTanTheta = abs(TanTheta(w));
        $if(!isinf(absTanTheta)) {
            // Compute _alpha_ for direction _w_
            auto alpha = sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
            auto alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
            return (-1.0f + sqrt(1.f + alpha2Tan2Theta)) * 0.5f;
        };
        return L;
    }

public:
    [[nodiscard]] TrowbridgeReitzDistribution(Float alphax, Float alphay) noexcept
        : alphax(alphax), alphay(alphay) {}

    // TrowbridgeReitzDistribution Public Methods
    [[nodiscard]] static inline Float RoughnessToAlpha(Float roughness) noexcept {
        roughness = max(roughness, 1e-3f);
        auto x = log(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x +
               0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
    }

    [[nodiscard]] Float G(Float3 wo, Float3 wi) const noexcept override {
        return 1.0f / (1.0f + Lambda(wo) + Lambda(wi));
    }

    [[nodiscard]] Float pdf(Float3 wo, Float3 wh) const noexcept override {
        return D(wh) * G1(wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
    }

    [[nodiscard]] Float D(Float3 wh) const noexcept override {
        auto d = def(0.0f);
        auto tan2Theta = Tan2Theta(wh);
        $if(!isinf(tan2Theta)) {
            auto cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
            auto e = (Cos2Phi(wh) / (alphax * alphax) +
                      Sin2Phi(wh) / (alphay * alphay)) *
                     tan2Theta;
            return 1.0f / (pi * alphax * alphay * cos4Theta * (1.0f + e) * (1.0f + e));
        };
        return d;
    }

    [[nodiscard]] Float3 sample_wh(Float3 wo, Float2 u) const noexcept override {
        auto s = sign(wo.z);
        auto wh = TrowbridgeReitzSample(s * wo, alphax, alphay, u.x, u.y);
        return s * wh;
    }
};

struct DisneyMicrofacetDistribution final : public TrowbridgeReitzDistribution {
    [[nodiscard]] explicit DisneyMicrofacetDistribution(Float alphax = 0.0f, Float alphay = 0.0f) noexcept
        : TrowbridgeReitzDistribution{alphax, alphay} {}
    [[nodiscard]] Float G(Float3 wo, Float3 wi) const noexcept override {
        // Disney uses the separable masking-shadowing model.
        return G1(wo) * G1(wi);
    }
};



}// namespace disney_impl

class DisneySurfaceClosure final : public Surface::Closure {

private:
public:
    DisneySurfaceClosure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> color, Expr<float> metallic, Expr<float> eta, Expr<float> roughness,
        Expr<float> specular_tint, Expr<float> anisotropic, Expr<float> sheen, Expr<float> sheen_tint,
        Expr<float> clearcoat, Expr<float> clearcoat_gloss, Expr<float> specular_trans,
        Expr<float> flatness, Expr<float> diffuse_trans) noexcept {
    }
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override;
    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override;
};

luisa::unique_ptr<Surface::Closure> DisneySurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<TextureHandle>(it.shape()->surface_buffer_id());
    return luisa::make_unique<DisneySurfaceClosure>(
        it, swl,
        pipeline.evaluate_color_texture(params.read(0), it, swl, time) /* color */,
        pipeline.evaluate_generic_texture(params.read(1), it, time).x /* metallic */,
        pipeline.evaluate_generic_texture(params.read(2), it, time).x /* eta */,
        pipeline.evaluate_generic_texture(params.read(3), it, time).x /* roughness */,
        pipeline.evaluate_generic_texture(params.read(4), it, time).x /* specular_tint */,
        pipeline.evaluate_generic_texture(params.read(5), it, time).x /* anisotropic */,
        pipeline.evaluate_generic_texture(params.read(6), it, time).x /* sheen */,
        pipeline.evaluate_generic_texture(params.read(7), it, time).x /* sheen_tint */,
        pipeline.evaluate_generic_texture(params.read(8), it, time).x /* clearcoat */,
        pipeline.evaluate_generic_texture(params.read(9), it, time).x /* clearcoat_gloss */,
        pipeline.evaluate_generic_texture(params.read(10), it, time).x /* specular_trans */,
        pipeline.evaluate_generic_texture(params.read(11), it, time).x /* flatness */,
        pipeline.evaluate_generic_texture(params.read(12), it, time).x /* diffuse_trans */);
}

Surface::Evaluation DisneySurfaceClosure::evaluate(Expr<float3> wi) const noexcept {
    return {};
}

Surface::Sample DisneySurfaceClosure::sample(Sampler::Instance &sampler) const noexcept {
    return {};
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DisneySurface)

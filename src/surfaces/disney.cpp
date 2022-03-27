//
// Created by Mike Smith on 2022/1/30.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

#include <utility>

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
    bool _thin;

public:
    DisneySurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _thin{desc->property_bool_or_default("thin", false)} {
#define LUISA_RENDER_DISNEY_PARAM_LOAD(name)                      \
    _##name = scene->load_texture(desc->property_node_or_default( \
        #name, SceneNodeDesc::shared_default_texture("Constant")));
        LUISA_RENDER_DISNEY_PARAM_LOAD(color)
        LUISA_RENDER_DISNEY_PARAM_LOAD(metallic)
        LUISA_RENDER_DISNEY_PARAM_LOAD(eta)
        LUISA_RENDER_DISNEY_PARAM_LOAD(roughness)
        LUISA_RENDER_DISNEY_PARAM_LOAD(specular_tint)
        LUISA_RENDER_DISNEY_PARAM_LOAD(anisotropic)
        LUISA_RENDER_DISNEY_PARAM_LOAD(sheen)
        LUISA_RENDER_DISNEY_PARAM_LOAD(sheen_tint)
        LUISA_RENDER_DISNEY_PARAM_LOAD(clearcoat)
        LUISA_RENDER_DISNEY_PARAM_LOAD(clearcoat_gloss)
        LUISA_RENDER_DISNEY_PARAM_LOAD(specular_trans)
        LUISA_RENDER_DISNEY_PARAM_LOAD(flatness)
        LUISA_RENDER_DISNEY_PARAM_LOAD(diffuse_trans)
#undef LUISA_RENDER_DISNEY_PARAM_LOAD
    }
    [[nodiscard]] auto thin() const noexcept { return _thin; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override {
        return LUISA_RENDER_PLUGIN_NAME;
    }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class DisneySurfaceInstance final : public Surface::Instance {

private:
    const Texture::Instance *_color{};
    const Texture::Instance *_metallic{};
    const Texture::Instance *_eta{};
    const Texture::Instance *_roughness{};
    const Texture::Instance *_specular_tint{};
    const Texture::Instance *_anisotropic{};
    const Texture::Instance *_sheen{};
    const Texture::Instance *_sheen_tint{};
    const Texture::Instance *_clearcoat{};
    const Texture::Instance *_clearcoat_gloss{};
    const Texture::Instance *_specular_trans{};
    const Texture::Instance *_flatness{};
    const Texture::Instance *_diffuse_trans{};

public:
    DisneySurfaceInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *color, const Texture::Instance *metallic,
        const Texture::Instance *eta, const Texture::Instance *roughness,
        const Texture::Instance *specular_tint, const Texture::Instance *anisotropic,
        const Texture::Instance *sheen, const Texture::Instance *sheen_tint,
        const Texture::Instance *clearcoat, const Texture::Instance *clearcoat_gloss,
        const Texture::Instance *specular_trans, const Texture::Instance *flatness,
        const Texture::Instance *diffuse_trans, bool thin) noexcept
        : Surface::Instance{pipeline, surface},
          _color{color}, _metallic{metallic}, _eta{eta},
          _roughness{roughness}, _specular_tint{specular_tint},
          _anisotropic{anisotropic}, _sheen{sheen}, _sheen_tint{sheen_tint},
          _clearcoat{clearcoat}, _clearcoat_gloss{clearcoat_gloss},
          _specular_trans{specular_trans}, _flatness{flatness},
          _diffuse_trans{diffuse_trans} {}
    [[nodiscard]] auto color() const { return _color; }
    [[nodiscard]] auto metallic() const { return _metallic; }
    [[nodiscard]] auto eta() const { return _eta; }
    [[nodiscard]] auto roughness() const { return _roughness; }
    [[nodiscard]] auto specular_tint() const { return _specular_tint; }
    [[nodiscard]] auto anisotropic() const { return _anisotropic; }
    [[nodiscard]] auto sheen() const { return _sheen; }
    [[nodiscard]] auto sheen_tint() const { return _sheen_tint; }
    [[nodiscard]] auto clearcoat() const { return _clearcoat; }
    [[nodiscard]] auto clearcoat_gloss() const { return _clearcoat_gloss; }
    [[nodiscard]] auto specular_trans() const { return _specular_trans; }
    [[nodiscard]] auto flatness() const { return _flatness; }
    [[nodiscard]] auto diffuse_trans() const { return _diffuse_trans; }
    [[nodiscard]] auto thin() const { return node<DisneySurface>()->thin(); }
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> DisneySurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto color = pipeline.build_texture(command_buffer, _color);
    auto metallic = pipeline.build_texture(command_buffer, _metallic);
    auto eta = pipeline.build_texture(command_buffer, _eta);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    auto specular_tint = pipeline.build_texture(command_buffer, _specular_tint);
    auto anisotropic = pipeline.build_texture(command_buffer, _anisotropic);
    auto sheen = pipeline.build_texture(command_buffer, _sheen);
    auto sheen_tint = pipeline.build_texture(command_buffer, _sheen_tint);
    auto clearcoat = pipeline.build_texture(command_buffer, _clearcoat);
    auto clearcoat_gloss = pipeline.build_texture(command_buffer, _clearcoat_gloss);
    auto specular_trans = pipeline.build_texture(command_buffer, _specular_trans);
    auto flatness = pipeline.build_texture(command_buffer, _flatness);
    auto diffuse_trans = pipeline.build_texture(command_buffer, _diffuse_trans);
    return luisa::make_unique<DisneySurfaceInstance>(
        pipeline, this,
        color, metallic, eta, roughness,
        specular_tint, anisotropic,
        sheen, sheen_tint,
        clearcoat, clearcoat_gloss,
        specular_trans, flatness,
        diffuse_trans, _thin);
}

using namespace compute;

namespace {

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
//
// The Schlick Fresnel approximation is:
//
// R = R(0) + (1 - R(0)) (1 - cos theta)^5,
//
// where R(0) is the reflectance at normal indicence.
[[nodiscard]] inline Float SchlickWeight(Expr<float> cosTheta) noexcept {
    auto m = saturate(1.f - cosTheta);
    return sqr(sqr(m)) * m;
}

[[nodiscard]] inline Float FrSchlick(Expr<float> R0, Expr<float> cosTheta) noexcept {
    return lerp(R0, 1.f, SchlickWeight(cosTheta));
}

// For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
// coming from air..
[[nodiscard]] inline Float SchlickR0FromEta(Float eta) {
    return sqr(eta - 1.f) / sqr(eta + 1.f);
}

class DisneyDiffuse final : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dR;
    };

private:
    SampledSpectrum R;

public:
    explicit DisneyDiffuse(SampledSpectrum R) noexcept : R{std::move(R)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto Fo = SchlickWeight(abs_cos_theta(wo));
        auto Fi = SchlickWeight(abs_cos_theta(wi));

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
        // Burley 2015, eq (4).
        return R * inv_pi * (1.f - Fo * .5f) * (1.f - Fi * .5f);
    }
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept {
        auto Fo = SchlickWeight(abs_cos_theta(wo));
        auto Fi = SchlickWeight(abs_cos_theta(wi));
        return {.dR = df * inv_pi * (1.f - Fo * .5f) * (1.f - Fi * .5f)};
    }
};

// "Fake" subsurface scattering lobe, based on the Hanrahan-Krueger BRDF
// approximation of the BSSRDF.
class DisneyFakeSS final : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dR;
        Float dRoughness;
    };

private:
    SampledSpectrum R;
    Float roughness;

public:
    DisneyFakeSS(SampledSpectrum R, Float roughness) noexcept
        : R{std::move(R)}, roughness{std::move(roughness)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto wh = wi + wo;
        auto valid = any(wh != 0.f);
        wh = normalize(wh);
        auto cosThetaD = dot(wi, wh);
        // Fss90 used to "flatten" retroreflection based on roughness
        auto Fss90 = cosThetaD * cosThetaD * roughness;
        auto Fo = SchlickWeight(abs_cos_theta(wo));
        auto Fi = SchlickWeight(abs_cos_theta(wi));
        auto Fss = lerp(1.0f, Fss90, Fo) * lerp(1.0f, Fss90, Fi);
        // 1.25 scale is used to (roughly) preserve albedo
        auto ss = 1.25f * (Fss * (1.f / (abs_cos_theta(wo) + abs_cos_theta(wi)) - .5f) + .5f);
        return R * ite(valid, inv_pi * ss, 0.f);
    }
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

class DisneyRetro final : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dR;
        Float dRoughness;
    };

private:
    SampledSpectrum R;
    Float roughness;

public:
    DisneyRetro(SampledSpectrum R, Float roughness) noexcept
        : R{std::move(R)}, roughness{std::move(roughness)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto wh = wi + wo;
        auto valid = any(wh != 0.f);
        wh = normalize(wh);
        auto cosThetaD = dot(wi, wh);
        auto Fo = SchlickWeight(abs_cos_theta(wo));
        auto Fi = SchlickWeight(abs_cos_theta(wi));
        auto Rr = 2.f * roughness * cosThetaD * cosThetaD;

        // Burley 2015, eq (4).
        auto f = ite(valid, inv_pi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1.f)), 0.f);
        return R * f;
    }
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

class DisneySheen final : public BxDF {

public:
    struct Gradient {
        SampledSpectrum dR;
    };

private:
    SampledSpectrum R;

public:
    explicit DisneySheen(SampledSpectrum R) noexcept : R{std::move(R)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto wh = wi + wo;
        auto valid = any(wh != 0.f);
        wh = normalize(wh);
        auto cosThetaD = dot(wi, wh);
        return R * ite(valid, SchlickWeight(cosThetaD), 0.f);
    }
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

[[nodiscard]] inline Float GTR1(Float cosTheta, Float alpha) noexcept {
    auto alpha2 = sqr(alpha);
    auto denom = pi * log(alpha2) * (1.f + (alpha2 - 1.f) * sqr(cosTheta));
    return (alpha2 - 1.f) / denom;
}

// Smith masking/shadowing term.
[[nodiscard]] inline Float smithG_GGX(Float cosTheta, Float alpha) noexcept {
    auto alpha2 = sqr(alpha);
    auto cosTheta2 = sqr(cosTheta);
    return 1.f / (cosTheta + sqrt(alpha2 + cosTheta2 - alpha2 * cosTheta2));
}

class DisneyClearcoat final {

public:
    struct Gradient {
        Float dWeight;
        Float dGloss;
    };

private:
    Float weight;
    Float gloss;

public:
    DisneyClearcoat(Float weight, Float gloss) noexcept
        : weight{std::move(weight)}, gloss{std::move(gloss)} {}
    [[nodiscard]] Float evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
        auto wh = wi + wo;
        auto valid = any(wh != 0.f);
        wh = normalize(wh);
        // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
        // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
        // (which is GTR2).
        auto Dr = GTR1(abs_cos_theta(wh), gloss);
        auto Fr = FrSchlick(.04f, dot(wo, wh));
        // The geometric term always based on alpha = 0.25.
        auto Gr = smithG_GGX(abs_cos_theta(wo), .25f) *
                  smithG_GGX(abs_cos_theta(wi), .25f);
        return ite(valid, weight * Gr * Fr * Dr * .25f, 0.f);
    }
    [[nodiscard]] Float sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *p) const noexcept {
        // TODO: double check all this: there still seem to be some very
        // occasional fireflies with clearcoat; presumably there is a bug
        // somewhere.
        auto alpha2 = gloss * gloss;
        auto cosTheta = sqrt(max(0.f, (1.f - pow(alpha2, 1.f - u[0])) / (1.f - alpha2)));
        auto sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        auto phi = 2.f * pi * u[1];
        auto wh = spherical_direction(sinTheta, cosTheta, phi);
        wh = ite(same_hemisphere(wo, wh), wh, -wh);
        *wi = reflect(wo, wh);
        auto valid = wo.z != 0.f & same_hemisphere(wo, *wi);
        *p = ite(valid, pdf(wo, *wi), 0.f);
        return ite(valid, evaluate(wo, *wi), 0.f);
    }
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept {
        auto wh = wi + wo;
        auto valid = same_hemisphere(wo, wi) & any(wh != 0.f);
        wh = normalize(wh);
        // The sampling routine samples wh exactly from the GTR1 distribution.
        // Thus, the final value of the PDF is just the value of the
        // distribution for wh converted to a mesure with respect to the
        // surface normal.
        auto Dr = GTR1(abs_cos_theta(wh), gloss);
        return ite(valid, Dr * abs_cos_theta(wh) / (4.f * dot(wo, wh)), 0.f);
    }
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

// Specialized Fresnel function used for the specular component, based on
// a mixture between dielectric and the Schlick Fresnel approximation.
class DisneyFresnel final : public Fresnel {

public:
    struct Gradient {
        SampledSpectrum dR0;
        Float dMetallic;
        Float dEta;
    };

private:
    SampledSpectrum R0;
    Float metallic;
    Float e;

public:
    DisneyFresnel(SampledSpectrum R0, Float metallic, Float eta) noexcept
        : R0{std::move(R0)}, metallic{std::move(metallic)}, e{std::move(eta)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosI) const noexcept override {
        auto fr = fresnel_dielectric(cosI, 1.f, e);
        return R0.map([&](auto, auto R) noexcept {
            return lerp(fr, FrSchlick(R, cosI), metallic);
        });
    }
    [[nodiscard]] auto &eta() const noexcept { return e; }
    [[nodiscard]] Gradient backward(Expr<float> cosThetaI, const SampledSpectrum &dFr) const noexcept {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

struct DisneyMicrofacetDistribution final : public TrowbridgeReitzDistribution {
    explicit DisneyMicrofacetDistribution(Expr<float2> alpha) noexcept
        : TrowbridgeReitzDistribution{alpha} {}
    [[nodiscard]] Float G(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        return G1(wo) * G1(wi);
    }
};

}// namespace

class DisneySurfaceClosure final : public Surface::Closure {

public:
    static constexpr auto refl_diffuse = 1u << 0u;
    static constexpr auto refl_fake_ss = 1u << 1u;
    static constexpr auto refl_retro = 1u << 2u;
    static constexpr auto refl_sheen = 1u << 3u;
    static constexpr auto refl_diffuse_like =
        refl_diffuse | refl_fake_ss |
        refl_retro | refl_sheen;
    static constexpr auto refl_specular = 1u << 4u;
    static constexpr auto refl_clearcoat = 1u << 5u;
    static constexpr auto trans_specular = 1u << 6u;
    static constexpr auto trans_thin_specular = 1u << 7u;
    static constexpr auto trans_thin_diffuse = 1u << 8u;

public:
    static constexpr std::array sampling_techniques{
        refl_diffuse_like, refl_specular, refl_clearcoat,
        trans_specular, trans_thin_specular, trans_thin_diffuse};
    static constexpr auto sampling_technique_diffuse = 0u;
    static constexpr auto sampling_technique_specular = 1u;
    static constexpr auto sampling_technique_clearcoat = 2u;
    static constexpr auto sampling_technique_specular_trans = 3u;
    static constexpr auto sampling_technique_thin_specular_trans = 4u;
    static constexpr auto sampling_technique_thin_diffuse_trans = 5u;
    static constexpr auto max_sampling_techique_count = 6u;
    static_assert(max_sampling_techique_count == std::size(sampling_techniques));

private:
    luisa::unique_ptr<DisneyDiffuse> _diffuse;
    luisa::unique_ptr<DisneyFakeSS> _fake_ss;
    luisa::unique_ptr<DisneyRetro> _retro;
    luisa::unique_ptr<DisneySheen> _sheen;
    luisa::unique_ptr<DisneyMicrofacetDistribution> _distrib;
    luisa::unique_ptr<DisneyFresnel> _fresnel;
    luisa::unique_ptr<MicrofacetReflection> _specular;
    luisa::unique_ptr<DisneyClearcoat> _clearcoat;
    luisa::unique_ptr<MicrofacetTransmission> _spec_trans;
    luisa::unique_ptr<TrowbridgeReitzDistribution> _thin_distrib;
    luisa::unique_ptr<MicrofacetTransmission> _thin_spec_trans;
    luisa::unique_ptr<LambertianTransmission> _diff_trans;
    UInt _lobes;
    Float _sampling_weights[max_sampling_techique_count];

public:
    DisneySurfaceClosure(
        const DisneySurfaceInstance *instance, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time,
        const SampledSpectrum &color, Expr<float> color_lum, Expr<float> metallic_in, Expr<float> eta_in, Expr<float> roughness,
        Expr<float> specular_tint, Expr<float> anisotropic, Expr<float> sheen, Expr<float> sheen_tint,
        Expr<float> clearcoat, Expr<float> clearcoat_gloss, Expr<float> specular_trans_in,
        Expr<float> flatness, Expr<float> diffuse_trans) noexcept
        : Surface::Closure{instance, it, swl, time}, _lobes{0u} {

        // TODO: should not generate lobes than are not used.
        constexpr auto black_threshold = 1e-6f;
        auto cos_theta_o = dot(_it.wo(), _it.shading().n());
        auto front_face = cos_theta_o > 0.f;
        auto metallic = ite(front_face, metallic_in, 0.f);
        auto specular_trans = (1.f - metallic) * specular_trans_in;
        auto diffuse_weight = (1.f - metallic) * (1.f - specular_trans);
        auto dt = diffuse_trans * .5f;// 0: all diffuse is reflected -> 1, transmitted
        auto Ctint_weight = ite(color_lum > 0.f, 1.f / color_lum, 1.f);
        auto Ctint = color * Ctint_weight;// normalize lum. to isolate hue+sat
        auto Ctint_lum = color_lum * Ctint_weight;
        auto eta = ite(eta_in < black_threshold, 1.5f, eta_in);
        auto thin = instance->thin();

        // diffuse
        auto diffuse_scale = ite(thin, (1.f - flatness) * (1.f - dt), ite(front_face, 1.f, 0.f));
        auto Cdiff_weight = diffuse_weight * diffuse_scale;
        auto Cdiff = color * Cdiff_weight;
        _diffuse = luisa::make_unique<DisneyDiffuse>(Cdiff);
        auto Cdiff_lum = color_lum * Cdiff_weight;
        _lobes |= ite(Cdiff_lum > black_threshold, refl_diffuse, 0u);
        auto Css_weight = ite(thin, diffuse_weight * flatness * (1.f - dt), 0.f);
        auto Css = Css_weight * color;
        _fake_ss = luisa::make_unique<DisneyFakeSS>(Css, roughness);
        auto Css_lum = color_lum * Css_weight;
        _lobes |= ite(Css_lum > black_threshold, refl_fake_ss, 0u);

        // retro-reflection
        auto Cretro_weight = ite(front_face | thin, diffuse_weight, 0.f);
        auto Cretro = Cretro_weight * color;
        _retro = luisa::make_unique<DisneyRetro>(Cretro, roughness);
        auto Cretro_lum = color_lum * Cretro_weight;
        _lobes |= ite(Cretro_lum > black_threshold, refl_retro, 0u);

        // sheen
        auto Csheen_weight = ite(front_face | thin, diffuse_weight * sheen, 0.f);
        auto Csheen = Csheen_weight * Ctint.map([sheen_tint](auto, auto t) noexcept {
            return lerp(1.f, t, sheen_tint);
        });
        _sheen = luisa::make_unique<DisneySheen>(Csheen);
        auto Csheen_lum = Csheen_weight * lerp(1.f, color_lum, specular_tint);
        _lobes |= ite(Csheen_lum > black_threshold, refl_sheen, 0u);

        // diffuse sampling weight
        _sampling_weights[sampling_technique_diffuse] = Cdiff_lum + Css_lum + Cretro_lum + Csheen_lum;

        // create the microfacet distribution for metallic and/or specular transmittance
        auto aspect = sqrt(1.f - anisotropic * .9f);
        auto alpha = make_float2(
            max(0.001f, sqr(roughness) / aspect),
            max(0.001f, sqr(roughness) * aspect));
        _distrib = luisa::make_unique<DisneyMicrofacetDistribution>(alpha);

        // specular is Trowbridge-Reitz with a modified Fresnel function
        auto SchlickR0 = SchlickR0FromEta(eta);
        auto Cspec0 = Ctint.map([&](auto i, auto t) noexcept {
            return lerp(lerp(1.f, t, specular_tint) * SchlickR0, color[i], metallic);
        });
        _fresnel = luisa::make_unique<DisneyFresnel>(Cspec0, metallic, eta);
        _specular = luisa::make_unique<MicrofacetReflection>(
            SampledSpectrum{swl.dimension(), 1.f}, _distrib.get(), _fresnel.get());
        _lobes |= refl_specular;// always consider the specular lobe

        // specular reflection sampling weight
        auto fr = fresnel_dielectric(cos_theta_o, 1.f, eta);
        auto F = clamp(lerp(fr, FrSchlick(1.f, cos_theta_o), metallic), 0.1f, 0.9f);
        auto Cspec0_lum = F * lerp(lerp(1.f, Ctint_lum, specular_tint) * SchlickR0, color_lum, metallic);
        _sampling_weights[sampling_technique_specular] = Cspec0_lum;

        // clearcoat
        auto cc = ite(front_face | thin, clearcoat, 0.f);
        auto gloss = lerp(.1f, .001f, clearcoat_gloss);
        _clearcoat = luisa::make_unique<DisneyClearcoat>(cc, gloss);
        _lobes |= ite(cc > black_threshold, refl_clearcoat, 0u);

        // clearcoat sampling weight
        _sampling_weights[sampling_technique_clearcoat] = cc * FrSchlick(.04f, 1.f);

        // specular transmission
        auto T = specular_trans * color.map([](auto, auto c) noexcept { return sqrt(c); });
        auto T_lum = specular_trans * sqrt(color_lum);
        auto Cst_weight = thin ? 0.f : 1.f;
        auto Cst = Cst_weight * T;
        _spec_trans = luisa::make_unique<MicrofacetTransmission>(
            Cst, _distrib.get(),
            SampledSpectrum{swl.dimension(), 1.f},
            SampledSpectrum{swl.dimension(), eta});
        auto Cst_lum = Cst_weight * T_lum;
        _lobes |= ite(Cst_lum > black_threshold, trans_specular, 0u);

        // specular transmission sampling weight
        _sampling_weights[sampling_technique_specular_trans] = (1.f - F) * Cst_lum;

        // thin specular transmission
        auto rscaled = (.65f * eta - .35f) * roughness;
        auto ascaled = make_float2(
            max(.001f, sqr(rscaled) / aspect),
            max(.001f, sqr(rscaled) * aspect));
        auto Ctst_weight = thin ? 1.f : 0.f;
        auto Ctst = Ctst_weight * T;
        _thin_distrib = luisa::make_unique<TrowbridgeReitzDistribution>(ascaled);
        _thin_spec_trans = luisa::make_unique<MicrofacetTransmission>(
            Ctst, _thin_distrib.get(),
            SampledSpectrum{swl.dimension(), 1.f},
            SampledSpectrum{swl.dimension(), eta});
        auto Ctst_lum = Ctst_weight * T_lum;
        _lobes |= ite(Ctst_lum > black_threshold, trans_thin_specular, 0u);

        // thin specular transmission sampling weight
        _sampling_weights[sampling_technique_thin_specular_trans] = (1.f - F) * Ctst_lum;

        // thin diffuse transmission
        auto Cdt_weight = ite(thin, dt, 0.f);
        auto Cdt = Cdt_weight * color;
        _diff_trans = luisa::make_unique<LambertianTransmission>(dt * color);
        auto Cdt_lum = Cdt_weight * color_lum;
        _lobes |= ite(Cdt_lum > black_threshold, trans_thin_diffuse, 0u);

        // thin diffuse transmission sampling weight
        _sampling_weights[sampling_technique_thin_diffuse_trans] = Cdt_lum;

        // normalize sampling weights
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < max_sampling_techique_count; i++) {
            auto regularized_weight = ite(
                (_lobes & sampling_techniques[i]) != 0u,
                sqrt(max(_sampling_weights[i], 1e-3f)), 0.f);
            _sampling_weights[i] = regularized_weight;
            sum_weights += regularized_weight;
        }
        auto inv_sum_weights = 1.f / sum_weights;
        for (auto &s : _sampling_weights) { s *= inv_sum_weights; }
    }
    [[nodiscard]] Surface::Evaluation evaluate_local(Float3 wo_local, Float3 wi_local) const noexcept {
        SampledSpectrum f{_swl.dimension()};
        auto pdf = def(0.f);
        // TODO: performance test
        $if(same_hemisphere(wo_local, wi_local)) {// reflection
            f = _specular->evaluate(wo_local, wi_local) +
                _diffuse->evaluate(wo_local, wi_local) +
                _fake_ss->evaluate(wo_local, wi_local) +
                _retro->evaluate(wo_local, wi_local) +
                _sheen->evaluate(wo_local, wi_local);
            pdf = _sampling_weights[sampling_technique_specular] *
                      _specular->pdf(wo_local, wi_local) +
                  _sampling_weights[sampling_technique_diffuse] *
                      _diffuse->pdf(wo_local, wi_local);
            $if((_lobes & refl_clearcoat) != 0u) {
                f += _clearcoat->evaluate(wo_local, wi_local);
                pdf += _sampling_weights[sampling_technique_clearcoat] *
                       _clearcoat->pdf(wo_local, wi_local);
            };
        }
        $else {// transmission
            $if((_lobes & trans_specular) != 0u) {
                f = _spec_trans->evaluate(wo_local, wi_local);
                pdf = _sampling_weights[sampling_technique_specular_trans] *
                      _spec_trans->pdf(wo_local, wi_local);
            }
            $else {
                f = _diff_trans->evaluate(wo_local, wi_local) +
                    _thin_spec_trans->evaluate(wo_local, wi_local);
                pdf = _sampling_weights[sampling_technique_thin_diffuse_trans] *
                          _diff_trans->pdf(wo_local, wi_local) +
                      _sampling_weights[sampling_technique_thin_specular_trans] *
                          _thin_spec_trans->pdf(wo_local, wi_local);
            };
        };
        auto thin = instance<DisneySurfaceInstance>()->thin();
        return {.f = f,
                .pdf = pdf,
                .alpha = _distrib->alpha(),
                .eta = SampledSpectrum{
                    _swl.dimension(),
                    ite(thin & wi_local.z < 0.f, _fresnel->eta(), 1.f)}};
    }
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        return evaluate_local(wo_local, wi_local);
    }
    [[nodiscard]] Surface::Sample sample(Expr<float> u_lobe, Expr<float2> u) const noexcept override {
        auto sampling_tech = def(0u);
        auto sum_weights = def(0.f);
        auto ux_remapped = def(0.f);
        auto lower_sum = def(0.f);
        auto upper_sum = def(1.f);
        for (auto i = 0u; i < max_sampling_techique_count; i++) {
            auto sel = (_lobes & sampling_techniques[i]) != 0u & (u_lobe > sum_weights);
            sampling_tech = ite(sel, i, sampling_tech);
            lower_sum = ite(sel, sum_weights, lower_sum);
            sum_weights += _sampling_weights[i];
            upper_sum = ite(sel, sum_weights, upper_sum);
        }

        // sample
        auto wo_local = _it.wo_local();
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto pdf = def(0.f);
        $switch(sampling_tech) {
            $case(0u) { static_cast<void>(_diffuse->sample(wo_local, &wi_local, u, &pdf)); };
            $case(1u) { static_cast<void>(_specular->sample(wo_local, &wi_local, u, &pdf)); };
            $case(2u) { static_cast<void>(_clearcoat->sample(wo_local, &wi_local, u, &pdf)); };
            $case(3u) { static_cast<void>(_spec_trans->sample(wo_local, &wi_local, u, &pdf)); };
            $case(4u) { static_cast<void>(_thin_spec_trans->sample(wo_local, &wi_local, u, &pdf)); };
            $case(5u) { static_cast<void>(_diff_trans->sample(wo_local, &wi_local, u, &pdf)); };
            $default { unreachable(); };
        };
        auto eval = evaluate_local(wo_local, wi_local);
        auto wi = _it.shading().local_to_world(wi_local);
        return {.wi = std::move(wi), .eval = std::move(eval)};
    }

    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

luisa::unique_ptr<Surface::Closure> DisneySurfaceInstance::closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto color_rgb = _color->evaluate(it, time).xyz();
    auto color_lum = srgb_to_cie_y(color_rgb);
    auto color = swl.albedo_from_srgb(color_rgb);
    auto metallic = _metallic ? _metallic->evaluate(it, time).x : 0.f;
    auto eta = _eta ? _eta->evaluate(it, time).x : 1.5f;
    auto roughness = _roughness ? _roughness->evaluate(it, time).x : 1.f;
    auto specular_tint = _specular_tint ? _specular_tint->evaluate(it, time).x : 1.f;
    auto anisotropic = _anisotropic ? _anisotropic->evaluate(it, time).x : 0.f;
    auto sheen = _sheen ? _sheen->evaluate(it, time).x : 0.f;
    auto sheen_tint = _sheen_tint ? _sheen_tint->evaluate(it, time).x : 0.f;
    auto clearcoat = _clearcoat ? _clearcoat->evaluate(it, time).x : 0.f;
    auto clearcoat_gloss = _clearcoat_gloss ? _clearcoat_gloss->evaluate(it, time).x : 0.f;
    auto specular_trans = _specular_trans ? _specular_trans->evaluate(it, time).x : 0.f;
    auto flatness = _flatness ? _flatness->evaluate(it, time).x : 0.f;
    auto diffuse_trans = _diffuse_trans ? _diffuse_trans->evaluate(it, time).x : 0.f;
    return luisa::make_unique<DisneySurfaceClosure>(
        this, it, swl, time, color, color_lum,
        metallic, eta, roughness, specular_tint, anisotropic,
        sheen, sheen_tint, clearcoat, clearcoat_gloss,
        specular_trans, flatness, diffuse_trans);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DisneySurface)

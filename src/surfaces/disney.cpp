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

class DisneySurface : public Surface {

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
    bool _remap_roughness;

public:
    DisneySurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _color{scene->load_texture(desc->property_node_or_default(
              "color", lazy_construct([desc] {
                  return desc->property_node_or_default("Kd");
              })))},
          _thin{desc->property_bool_or_default("thin", false)},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
#define LUISA_RENDER_DISNEY_PARAM_LOAD(name) \
    _##name = scene->load_texture(desc->property_node_or_default(#name));
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
    [[nodiscard]] luisa::string_view impl_type() const noexcept override {
        return LUISA_RENDER_PLUGIN_NAME;
    }
    [[nodiscard]] uint properties() const noexcept override {
        auto properties = property_reflective;
        if (_thin) {
            if ((_specular_trans != nullptr && !_specular_trans->is_black()) ||
                (_diffuse_trans != nullptr && !_diffuse_trans->is_black())) {
                properties |= property_thin;
            }
        } else {
            if (_specular_trans != nullptr && !_specular_trans->is_black()) {
                properties |= property_transmissive;
            }
        }
        return properties;
    }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class DisneySurfaceInstance : public Surface::Instance {

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
        const Texture::Instance *diffuse_trans) noexcept
        : Surface::Instance{pipeline, surface},
          _color{color}, _metallic{metallic}, _eta{eta},
          _roughness{roughness}, _specular_tint{specular_tint},
          _anisotropic{anisotropic}, _sheen{sheen}, _sheen_tint{sheen_tint},
          _clearcoat{clearcoat}, _clearcoat_gloss{clearcoat_gloss},
          _specular_trans{specular_trans}, _flatness{flatness},
          _diffuse_trans{diffuse_trans} {}
    [[nodiscard]] auto color() const noexcept { return _color; }
    [[nodiscard]] auto metallic() const noexcept { return _metallic; }
    [[nodiscard]] auto eta() const noexcept { return _eta; }
    [[nodiscard]] auto roughness() const noexcept { return _roughness; }
    [[nodiscard]] auto specular_tint() const noexcept { return _specular_tint; }
    [[nodiscard]] auto anisotropic() const noexcept { return _anisotropic; }
    [[nodiscard]] auto sheen() const noexcept { return _sheen; }
    [[nodiscard]] auto sheen_tint() const noexcept { return _sheen_tint; }
    [[nodiscard]] auto clearcoat() const noexcept { return _clearcoat; }
    [[nodiscard]] auto clearcoat_gloss() const noexcept { return _clearcoat_gloss; }
    [[nodiscard]] auto specular_trans() const noexcept { return _specular_trans; }
    [[nodiscard]] auto flatness() const noexcept { return _flatness; }
    [[nodiscard]] auto diffuse_trans() const noexcept { return _diffuse_trans; }

public:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
        Expr<float> eta, Expr<float> time) const noexcept override;
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
    auto diffuse_trans = is_thin() ? pipeline.build_texture(command_buffer, _diffuse_trans) : nullptr;
    return luisa::make_unique<DisneySurfaceInstance>(
        pipeline, this,
        color, metallic, eta, roughness,
        specular_tint, anisotropic,
        sheen, sheen_tint,
        clearcoat, clearcoat_gloss,
        specular_trans, flatness,
        diffuse_trans);
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
    return sqr((eta - 1.f) / (eta + 1.f));
}

class DisneyDiffuse final : public BxDF {

private:
    SampledSpectrum R;

public:
    explicit DisneyDiffuse(const SampledSpectrum &R) noexcept : R{R} {}
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return R; }
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        static Callable impl = [](Float3 wo, Float3 wi) noexcept {
            auto Fo = SchlickWeight(abs_cos_theta(wo));
            auto Fi = SchlickWeight(abs_cos_theta(wi));
            // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
            // Burley 2015, eq (4).
            return inv_pi * (1.f - Fo * .5f) * (1.f - Fi * .5f);
        };
        return R * impl(wo, wi);
    }
};

// "Fake" subsurface scattering lobe, based on the Hanrahan-Krueger BRDF
// approximation of the BSSRDF.
class DisneyFakeSS final : public BxDF {

private:
    SampledSpectrum R;
    Float roughness;

public:
    DisneyFakeSS(const SampledSpectrum &R, Float roughness) noexcept
        : R{R}, roughness{std::move(roughness)} {}
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return R; }
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        static Callable impl = [](Float3 wo, Float3 wi, Float roughness) noexcept {
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
            return ite(valid, inv_pi * ss, 0.f);
        };
        return R * impl(wo, wi, roughness);
    }
};

class DisneyRetro final : public BxDF {

private:
    SampledSpectrum R;
    Float roughness;

public:
    DisneyRetro(const SampledSpectrum &R, Float roughness) noexcept
        : R{R}, roughness{std::move(roughness)} {}
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return R; }
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        static Callable impl = [](Float3 wo, Float3 wi, Float roughness) noexcept {
            auto wh = wi + wo;
            auto valid = any(wh != 0.f);
            wh = normalize(wh);
            auto cosThetaD = dot(wi, wh);
            auto Fo = SchlickWeight(abs_cos_theta(wo));
            auto Fi = SchlickWeight(abs_cos_theta(wi));
            auto Rr = 2.f * roughness * cosThetaD * cosThetaD;
            // Burley 2015, eq (4).
            return ite(valid, inv_pi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1.f)), 0.f);
        };
        return R * impl(wo, wi, roughness);
    }
};

class DisneySheen final : public BxDF {

private:
    SampledSpectrum R;

public:
    explicit DisneySheen(const SampledSpectrum &R) noexcept : R{R} {}
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return R; }
    [[nodiscard]] SampledSpectrum evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        static Callable impl = [](Float3 wo, Float3 wi) noexcept {
            auto wh = wi + wo;
            auto valid = any(wh != 0.f);
            wh = normalize(wh);
            auto cosThetaD = dot(wi, wh);
            return ite(valid, SchlickWeight(cosThetaD), 0.f);
        };
        return R * impl(wo, wi);
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

private:
    Float weight;
    Float gloss;

public:
    DisneyClearcoat(Float weight, Float gloss) noexcept
        : weight{std::move(weight)}, gloss{std::move(gloss)} {}
    [[nodiscard]] Float evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept {
        static Callable impl = [](Float3 wo, Float3 wi, Float weight, Float gloss) noexcept {
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
        };
        return impl(wo, wi, weight, gloss);
    }
    [[nodiscard]] BxDF::SampledDirection sample_wi(Expr<float3> wo, Expr<float2> u) const noexcept {
        static Callable impl = [](Float3 wo, Float2 u, Float gloss) noexcept {
            // TODO: double check all this: there still seem to be some very
            // occasional fireflies with clearcoat; presumably there is a bug
            // somewhere.
            auto alpha2 = gloss * gloss;
            auto cosTheta = sqrt(max(0.f, (1.f - pow(alpha2, 1.f - u[0])) / (1.f - alpha2)));
            auto sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
            auto phi = 2.f * pi * u[1];
            auto wh = spherical_direction(sinTheta, cosTheta, phi);
            wh = ite(same_hemisphere(wo, wh), wh, -wh);
            return reflect(wo, wh);
        };
        auto wi = impl(wo, u, gloss);
        return {.wi = wi, .valid = same_hemisphere(wo, wi)};
    }
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept {
        static Callable impl = [](Float3 wo, Float3 wi, Float gloss) noexcept {
            auto wh = wi + wo;
            auto valid = same_hemisphere(wo, wi) & any(wh != 0.f);
            wh = normalize(wh);
            // The sampling routine samples wh exactly from the GTR1 distribution.
            // Thus, the final value of the PDF is just the value of the
            // distribution for wh converted to a mesure with respect to the
            // surface normal.
            auto Dr = GTR1(abs_cos_theta(wh), gloss);
            return ite(valid, Dr * abs_cos_theta(wh) / (4.f * dot(wo, wh)), 0.f);
        };
        return impl(wo, wi, gloss);
    }
};

// Specialized Fresnel function used for the specular component, based on
// a mixture between dielectric and the Schlick Fresnel approximation.
class DisneyFresnel final : public Fresnel {

private:
    SampledSpectrum R0;
    Float metallic;
    Float e;
    bool two_sided;

public:
    DisneyFresnel(const SampledSpectrum &R0, Float metallic, Float eta, bool two_sided) noexcept
        : R0{R0}, metallic{std::move(metallic)}, e{std::move(eta)}, two_sided{two_sided} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosI_in) const noexcept override {
        auto cosI = two_sided ? abs(cosI_in) : def(cosI_in);
        auto fr = fresnel_dielectric(cosI, 1.f, e);
        auto f0 = R0.map([cosI](auto x) noexcept { return FrSchlick(x, cosI); });
        return lerp(fr, f0, metallic);
    }
    [[nodiscard]] auto &eta() const noexcept { return e; }
};

struct DisneyMicrofacetDistribution final : public TrowbridgeReitzDistribution {
    explicit DisneyMicrofacetDistribution(Expr<float2> alpha) noexcept
        : TrowbridgeReitzDistribution{alpha} {}
};

}// namespace

// pImpl idiom.
class DisneyClosureImplBase {

public:
    virtual ~DisneyClosureImplBase() noexcept = default;
    [[nodiscard]] virtual SampledSpectrum albedo() const noexcept = 0;
    [[nodiscard]] virtual Float2 roughness() const noexcept = 0;
    [[nodiscard]] virtual luisa::optional<Float> eta() const noexcept = 0;
    // Note: evaluation and sample should rely on the back pointer to the closure
    [[nodiscard]] virtual Surface::Evaluation evaluate(
        const Surface::Closure *cls, Expr<float3> wo,
        Expr<float3> wi, TransportMode mode) const noexcept = 0;
    [[nodiscard]] virtual Surface::Sample sample(
        const Surface::Closure *cls, Expr<float3> wo,
        Expr<float> u_lobe, Expr<float2> u, TransportMode mode) const noexcept = 0;
};

class DisneySurfaceClosureImpl final : public DisneyClosureImplBase {

public:
    static constexpr auto max_sampling_techique_count = 4u;

private:
    SampledSpectrum _color;
    luisa::unique_ptr<DisneyDiffuse> _diffuse;
    luisa::unique_ptr<DisneyFakeSS> _fake_ss;
    luisa::unique_ptr<DisneyRetro> _retro;
    luisa::unique_ptr<DisneySheen> _sheen;
    luisa::unique_ptr<DisneyMicrofacetDistribution> _distrib;
    luisa::unique_ptr<DisneyFresnel> _fresnel;
    luisa::unique_ptr<MicrofacetReflection> _specular;
    luisa::unique_ptr<DisneyClearcoat> _clearcoat;
    luisa::unique_ptr<MicrofacetTransmission> _spec_trans;
    Float _eta_t;
    Float _sampling_weights[max_sampling_techique_count];
    uint _sampling_technique_count{0u};
    uint _diffuse_like_technique_index{~0u};
    uint _specular_technique_index{~0u};
    uint _clearcoat_technique_index{~0u};
    uint _spec_trans_technique_index{~0u};

public:
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _color; }
    [[nodiscard]] Float2 roughness() const noexcept override {
        return DisneyMicrofacetDistribution::alpha_to_roughness(_distrib->alpha());
    }
    DisneySurfaceClosureImpl(const Surface::Closure *cls, Expr<float> eta_i,
                             const Texture::Instance *color_tex, const Texture::Instance *metallic_tex,
                             const Texture::Instance *eta_tex, const Texture::Instance *roughness_tex,
                             const Texture::Instance *spec_tint_tex, const Texture::Instance *aniso_tex,
                             const Texture::Instance *sheen_tex, const Texture::Instance *sheen_tint_tex,
                             const Texture::Instance *clearcoat_tex, const Texture::Instance *clearcoat_gloss_tex,
                             const Texture::Instance *spec_trans_tex, const Texture::Instance *flatness_tex) noexcept
        : _color{cls->swl().dimension()} {
        auto color_decode = color_tex ?
                                color_tex->evaluate_albedo_spectrum(*cls->it(), cls->swl(), cls->time()) :
                                Spectrum::Decode::one(cls->swl().dimension());
        _color = color_decode.value;
        auto color_lum = color_decode.strength;
        auto metallic = metallic_tex ? metallic_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        auto specular_trans = spec_trans_tex ? spec_trans_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        auto diffuse_weight = (1.f - metallic) * (1.f - specular_trans);
        auto flatness = flatness_tex ? flatness_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        auto roughness = roughness_tex ? roughness_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        if (cls->instance()->node<DisneySurface>()->remap_roughness()) {
            roughness = DisneyMicrofacetDistribution::roughness_to_alpha(roughness);
        }
        auto tint_weight = ite(color_lum > 0.f, 1.f / color_lum, 1.f);
        auto tint = saturate(_color * tint_weight);// normalize lum. to isolate hue+sat
        auto tint_lum = color_lum * tint_weight;

        static constexpr auto black_threshold = 1e-4f;

        // diffuse-like lobes: diffuse, retro-reflection, and optionally fake subsurface scattering and sheen
        if (color_tex == nullptr || !color_tex->node()->is_black()) {
            auto Cdiff_weight = diffuse_weight * (1.f - flatness);
            auto Cdiff = _color * Cdiff_weight;
            _diffuse = luisa::make_unique<DisneyDiffuse>(Cdiff);
            _retro = luisa::make_unique<DisneyRetro>(Cdiff, roughness);
            // fake subsurface scattering
            if (flatness_tex && !flatness_tex->node()->is_black()) {
                auto Css_weight = diffuse_weight * flatness;
                auto Css = Css_weight * _color;
                _fake_ss = luisa::make_unique<DisneyFakeSS>(Css, roughness);
            }
            auto sampling_weight = diffuse_weight * color_lum;
            // sheen
            if (sheen_tex && !sheen_tex->node()->is_black()) {
                auto sheen = sheen_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x;
                auto sheen_tint = sheen_tint_tex ? sheen_tint_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
                auto Csheen_weight = diffuse_weight * sheen;
                auto Csheen = Csheen_weight * lerp(1.f, tint, sheen_tint);
                _sheen = luisa::make_unique<DisneySheen>(Csheen);
                auto sheen_lum = Csheen_weight * lerp(1.f, tint_lum, sheen_tint);
                sampling_weight += sheen_lum * .1f;
            }
            _diffuse_like_technique_index = _sampling_technique_count++;
            _sampling_weights[_diffuse_like_technique_index] = saturate(sampling_weight);
        }

        // specular lobes: clearcoat, microfacet reflection, and optionally microfacet transmission
        auto spec_tint = spec_tint_tex ? spec_tint_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        _eta_t = eta_tex ? eta_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 1.5f;
        auto eta = _eta_t / eta_i;
        // specular is Trowbridge-Reitz with a modified Fresnel function
        auto SchlickR0 = SchlickR0FromEta(eta);
        auto Cspec0 = lerp(lerp(1.f, tint, spec_tint) * SchlickR0, _color, metallic);
        _fresnel = luisa::make_unique<DisneyFresnel>(
            Cspec0, metallic, eta, !cls->instance()->node()->is_transmissive());

        // create the microfacet distribution for metallic and/or specular transmittance
        auto aniso = aniso_tex ? aniso_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        auto aspect = sqrt(1.f - aniso * .9f);
        auto alpha = make_float2(max(0.001f, roughness / aspect),
                                 max(0.001f, roughness * aspect));
        _distrib = luisa::make_unique<DisneyMicrofacetDistribution>(alpha);
        _specular = luisa::make_unique<MicrofacetReflection>(
            SampledSpectrum{cls->swl().dimension(), 1.f}, _distrib.get(), _fresnel.get());
        // specular reflection sampling weight
        auto Cspec0_lum = lerp(lerp(1.f, tint_lum, spec_tint) * SchlickR0, color_lum, metallic);
        _specular_technique_index = _sampling_technique_count++;
        _sampling_weights[_specular_technique_index] = saturate(Cspec0_lum);

        // clearcoat
        if (clearcoat_tex && !clearcoat_tex->node()->is_black()) {
            auto cc = clearcoat_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x;
            auto gloss = lerp(.1f, .001f, clearcoat_gloss_tex ? clearcoat_gloss_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 1.f);
            _clearcoat = luisa::make_unique<DisneyClearcoat>(cc, gloss);
            // clearcoat sampling weight
            _clearcoat_technique_index = _sampling_technique_count++;
            _sampling_weights[_clearcoat_technique_index] = saturate(cc * FrSchlick(.04f, 1.f));
        }

        // specular transmission
        if (spec_trans_tex && !spec_trans_tex->node()->is_black()) {
            auto Cst_weight = (1.f - metallic) * specular_trans;
            auto Cst = Cst_weight * sqrt(_color);
            _spec_trans = luisa::make_unique<MicrofacetTransmission>(
                Cst, _distrib.get(), eta_i, _eta_t);
            auto Cst_lum = Cst_weight * sqrt(color_lum);
            // specular transmission sampling weight
            _spec_trans_technique_index = _sampling_technique_count++;
            _sampling_weights[_spec_trans_technique_index] = saturate(Cst_lum);
        }

        // normalize sampling weights
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            sum_weights += _sampling_weights[i];
        }
        auto inv_sum_weights = ite(sum_weights == 0.f, 0.f, 1.f / sum_weights);
        for (auto &s : _sampling_weights) { s *= inv_sum_weights; }
    }
    [[nodiscard]] luisa::optional<Float> eta() const noexcept override {
        if (_spec_trans == nullptr) { return luisa::nullopt; }
        return _eta_t;
    }

private:
    [[nodiscard]] Surface::Evaluation _evaluate_local(const Surface::Closure *cls,
                                                      Expr<float3> wo, Expr<float3> wi,
                                                      Expr<float3> wo_local, Expr<float3> wi_local,
                                                      TransportMode mode) const noexcept {
        SampledSpectrum f{cls->swl().dimension(), 0.f};
        auto pdf = def(0.f);
        auto cos_theta_i = def(0.f);
        $if(same_hemisphere(wo_local, wi_local)) {// reflection
            if (_diffuse) {
                f += _diffuse->evaluate(wo_local, wi_local, mode);
                f += _retro->evaluate(wo_local, wi_local, mode);
                if (_fake_ss) { f += _fake_ss->evaluate(wo_local, wi_local, mode); }
                if (_sheen) { f += _sheen->evaluate(wo_local, wi_local, mode); }
                pdf += _sampling_weights[_diffuse_like_technique_index] *
                       _diffuse->pdf(wo_local, wi_local, mode);
            }
            if (_specular) {
                f += _specular->evaluate(wo_local, wi_local, mode);
                pdf += _sampling_weights[_specular_technique_index] *
                       _specular->pdf(wo_local, wi_local, mode);
            }
            if (_clearcoat) {
                f += _clearcoat->evaluate(wo_local, wi_local);
                pdf += _sampling_weights[_clearcoat_technique_index] *
                       _clearcoat->pdf(wo_local, wi_local);
            }
            cos_theta_i = ite(cls->it()->shape()->shadow_terminator_factor() > 0.f |
                                  cls->it()->same_sided(wo, wi),
                              abs_cos_theta(wi_local), 0.f);
        }
        $else {// transmission
            if (_spec_trans) {
                f = _spec_trans->evaluate(wo_local, wi_local, mode);
                pdf = _sampling_weights[_spec_trans_technique_index] *
                      _spec_trans->pdf(wo_local, wi_local, mode);
            }
            cos_theta_i = abs_cos_theta(wi_local);
        };
        return {.f = f * cos_theta_i, .pdf = pdf};
    }
    [[nodiscard]] Surface::Evaluation evaluate(const Surface::Closure *cls,
                                               Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode) const noexcept override {
        auto wo_local = cls->it()->shading().world_to_local(wo);
        auto wi_local = cls->it()->shading().world_to_local(wi);
        return _evaluate_local(cls, wo, wi, wo_local, wi_local, mode);
    }
    [[nodiscard]] Surface::Sample sample(const Surface::Closure *cls,
                                         Expr<float3> wo, Expr<float> u_lobe,
                                         Expr<float2> u, TransportMode mode) const noexcept override {
        auto sampling_tech = def(0u);
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            sampling_tech = ite(u_lobe > sum_weights, i, sampling_tech);
            sum_weights += _sampling_weights[i];
        }
        // sample
        auto wo_local = cls->it()->shading().world_to_local(wo);
        auto event = def(Surface::event_reflect);
        BxDF::SampledDirection wi_sample;
        $switch(sampling_tech) {
            if (_diffuse) {
                $case(_diffuse_like_technique_index) {
                    wi_sample = _diffuse->sample_wi(wo_local, u, mode);
                };
            }
            if (_specular) {
                $case(_specular_technique_index) {
                    wi_sample = _specular->sample_wi(wo_local, u, mode);
                };
            }
            if (_clearcoat) {
                $case(_clearcoat_technique_index) {
                    wi_sample = _clearcoat->sample_wi(wo_local, u);
                };
            }
            if (_spec_trans) {
                $case(_spec_trans_technique_index) {
                    wi_sample = _spec_trans->sample_wi(wo_local, u, mode);
                    event = ite(cos_theta(wo_local) > 0.f, Surface::event_enter, Surface::event_exit);
                };
            }
            $default { unreachable(); };
        };
        auto eval = Surface::Evaluation::zero(_color.dimension());
        auto wi = cls->it()->shading().local_to_world(wi_sample.wi);
        $if(wi_sample.valid) {
            eval = _evaluate_local(cls, wo, wi, wo_local, wi_sample.wi, mode);
        };
        return {.eval = eval, .wi = wi, .event = event};
    }
};

class ThinDisneySurfaceClosureImpl : public DisneyClosureImplBase {

public:
    static constexpr auto max_sampling_techique_count = 5u;

private:
    SampledSpectrum _color;
    luisa::unique_ptr<DisneyDiffuse> _diffuse;
    luisa::unique_ptr<DisneyFakeSS> _fake_ss;
    luisa::unique_ptr<DisneyRetro> _retro;
    luisa::unique_ptr<DisneySheen> _sheen;
    luisa::unique_ptr<DisneyMicrofacetDistribution> _distrib;
    luisa::unique_ptr<DisneyFresnel> _fresnel;
    luisa::unique_ptr<MicrofacetReflection> _specular;
    luisa::unique_ptr<DisneyClearcoat> _clearcoat;
    luisa::unique_ptr<MicrofacetTransmission> _spec_trans;
    luisa::unique_ptr<LambertianTransmission> _diff_trans;
    luisa::unique_ptr<TrowbridgeReitzDistribution> _thin_distrib;
    Float _eta_t;
    Float _sampling_weights[max_sampling_techique_count];
    uint _sampling_technique_count{0u};
    uint _diffuse_like_technique_index{~0u};
    uint _specular_technique_index{~0u};
    uint _clearcoat_technique_index{~0u};
    uint _spec_trans_technique_index{~0u};
    uint _diff_trans_technique_index{~0u};

public:
    ThinDisneySurfaceClosureImpl(const Surface::Closure *cls, Expr<float> eta_i,
                                 const Texture::Instance *color_tex, const Texture::Instance *metallic_tex,
                                 const Texture::Instance *eta_tex, const Texture::Instance *roughness_tex,
                                 const Texture::Instance *spec_tint_tex, const Texture::Instance *aniso_tex,
                                 const Texture::Instance *sheen_tex, const Texture::Instance *sheen_tint_tex,
                                 const Texture::Instance *clearcoat_tex, const Texture::Instance *clearcoat_gloss_tex,
                                 const Texture::Instance *spec_trans_tex, const Texture::Instance *flatness_tex,
                                 const Texture::Instance *diff_trans_tex) noexcept
        : _color{cls->swl().dimension()} {
        auto color_decode = color_tex ?
                                color_tex->evaluate_albedo_spectrum(*cls->it(), cls->swl(), cls->time()) :
                                Spectrum::Decode::one(cls->swl().dimension());
        _color = color_decode.value;
        auto color_lum = color_decode.strength;
        auto metallic = metallic_tex ? metallic_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        auto specular_trans = spec_trans_tex ? spec_trans_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        auto flatness = flatness_tex ? flatness_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        auto roughness = roughness_tex ? roughness_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : .5f;
        if (cls->instance()->node<DisneySurface>()->remap_roughness()) {
            roughness = DisneyMicrofacetDistribution::roughness_to_alpha(roughness);
        }
        auto diffuse_trans = diff_trans_tex ? diff_trans_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x * .5f : 0.f;
        auto diffuse_weight = (1.f - metallic) * (1.f - specular_trans);
        auto diff_refl_weight = diffuse_weight * (1.f - diffuse_trans);
        auto diff_trans_weight = diffuse_weight * diffuse_trans;
        auto tint_weight = ite(color_lum > 0.f, 1.f / color_lum, 1.f);
        auto tint = saturate(_color * tint_weight);// normalize lum. to isolate hue+sat
        auto tint_lum = color_lum * tint_weight;

        static constexpr auto black_threshold = 1e-4f;

        // diffuse-like lobes: diffuse, retro-reflection, and optionally fake subsurface scattering and sheen
        if (color_tex == nullptr || !color_tex->node()->is_black()) {
            auto Cdiff_weight = diff_refl_weight * (1.f - flatness);
            auto Cdiff = _color * Cdiff_weight;
            _diffuse = luisa::make_unique<DisneyDiffuse>(Cdiff);
            _retro = luisa::make_unique<DisneyRetro>(Cdiff, roughness);
            // fake subsurface scattering
            if (flatness_tex && !flatness_tex->node()->is_black()) {
                auto Css_weight = diff_refl_weight * flatness * (1.f - diffuse_trans);
                auto Css = Css_weight * _color;
                _fake_ss = luisa::make_unique<DisneyFakeSS>(Css, roughness);
            }
            auto sampling_weight = diff_refl_weight * color_lum;
            // sheen
            if (sheen_tex && !sheen_tex->node()->is_black()) {
                auto sheen = sheen_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x;
                auto sheen_tint = sheen_tint_tex ? sheen_tint_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
                auto Csheen_weight = diff_refl_weight * sheen * (1.f - diffuse_trans);
                auto Csheen = Csheen_weight * lerp(1.f, tint, sheen_tint);
                _sheen = luisa::make_unique<DisneySheen>(Csheen);
                auto sheen_lum = Csheen_weight * lerp(1.f, tint_lum, sheen_tint);
                sampling_weight += sheen_lum * .1f;
            }
            _diffuse_like_technique_index = _sampling_technique_count++;
            _sampling_weights[_diffuse_like_technique_index] = saturate(sampling_weight);
        }

        // specular lobes: clearcoat, microfacet reflection, and optionally microfacet transmission
        auto spec_tint = spec_tint_tex ? spec_tint_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        _eta_t = eta_tex ? eta_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 1.5f;
        auto eta = _eta_t / eta_i;
        // specular is Trowbridge-Reitz with a modified Fresnel function
        auto SchlickR0 = SchlickR0FromEta(eta);
        auto Cspec0 = lerp(lerp(1.f, tint, spec_tint) * SchlickR0, _color, metallic);
        _fresnel = luisa::make_unique<DisneyFresnel>(
            Cspec0, metallic, eta, !cls->instance()->node()->is_transmissive());

        // create the microfacet distribution for metallic and/or specular transmittance
        auto aniso = aniso_tex ? aniso_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 0.f;
        auto aspect = sqrt(1.f - aniso * .9f);
        auto alpha = make_float2(max(0.001f, roughness / aspect),
                                 max(0.001f, roughness * aspect));
        _distrib = luisa::make_unique<DisneyMicrofacetDistribution>(alpha);
        _specular = luisa::make_unique<MicrofacetReflection>(
            SampledSpectrum{cls->swl().dimension(), 1.f}, _distrib.get(), _fresnel.get());
        // specular reflection sampling weight
        auto Cspec0_lum = lerp(lerp(1.f, tint_lum, spec_tint) * SchlickR0, color_lum, metallic);
        _specular_technique_index = _sampling_technique_count++;
        _sampling_weights[_specular_technique_index] = saturate(Cspec0_lum);

        // clearcoat
        if (clearcoat_tex && !clearcoat_tex->node()->is_black()) {
            auto cc = clearcoat_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x;
            auto gloss = lerp(.1f, .001f, clearcoat_gloss_tex ? clearcoat_gloss_tex->evaluate(*cls->it(), cls->swl(), cls->time()).x : 1.f);
            _clearcoat = luisa::make_unique<DisneyClearcoat>(cc, gloss);
            // clearcoat sampling weight
            _clearcoat_technique_index = _sampling_technique_count++;
            _sampling_weights[_clearcoat_technique_index] = saturate(cc * FrSchlick(.04f, 1.f));
        }

        // specular transmission
        if (spec_trans_tex && !spec_trans_tex->node()->is_black()) {
            // thin specular transmission distribution
            auto rscaled = (.65f * eta - .35f) * roughness;
            auto ascaled = make_float2(max(.001f, rscaled / aspect),
                                       max(.001f, rscaled * aspect));
            _thin_distrib = luisa::make_unique<TrowbridgeReitzDistribution>(ascaled);
            auto Cst_weight = (1.f - metallic) * specular_trans;
            auto Cst = Cst_weight * _color;
            _spec_trans = luisa::make_unique<MicrofacetTransmission>(
                Cst, _thin_distrib.get(), eta_i, _eta_t);
            auto Cst_lum = Cst_weight * color_lum;
            // specular transmission sampling weight
            _spec_trans_technique_index = _sampling_technique_count++;
            _sampling_weights[_spec_trans_technique_index] = saturate(Cst_lum);
        }

        // diffuse transmission
        if (diff_trans_tex && !diff_trans_tex->node()->is_black()) {
            auto Cdt = diff_trans_weight * _color;
            auto Cdt_lum = diff_trans_weight * color_lum;
            _diff_trans = luisa::make_unique<LambertianTransmission>(Cdt);
            _diff_trans_technique_index = _sampling_technique_count++;
            _sampling_weights[_diff_trans_technique_index] = saturate(Cdt_lum);
        }

        // normalize sampling weights
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            sum_weights += _sampling_weights[i];
        }
        auto inv_sum_weights = ite(sum_weights == 0.f, 0.f, 1.f / sum_weights);
        for (auto &s : _sampling_weights) { s *= inv_sum_weights; }
    }
    [[nodiscard]] luisa::optional<Float> eta() const noexcept override { return luisa::nullopt; }

private:
    [[nodiscard]] Surface::Evaluation _evaluate_local(const Surface::Closure *cls,
                                                      Expr<float3> wo, Expr<float3> wi,
                                                      Expr<float3> wo_local, Expr<float3> wi_local,
                                                      TransportMode mode) const noexcept {
        SampledSpectrum f{cls->swl().dimension(), 0.f};
        auto pdf = def(0.f);
        auto cos_theta_i = def(1.f);
        $if(same_hemisphere(wo_local, wi_local)) {// reflection
            if (_diffuse) {
                f += _diffuse->evaluate(wo_local, wi_local, mode);
                f += _retro->evaluate(wo_local, wi_local, mode);
                if (_fake_ss) { f += _fake_ss->evaluate(wo_local, wi_local, mode); }
                if (_sheen) { f += _sheen->evaluate(wo_local, wi_local, mode); }
                pdf += _sampling_weights[_diffuse_like_technique_index] *
                       _diffuse->pdf(wo_local, wi_local, mode);
            }
            if (_specular) {
                f += _specular->evaluate(wo_local, wi_local, mode);
                pdf += _sampling_weights[_specular_technique_index] *
                       _specular->pdf(wo_local, wi_local, mode);
            }
            if (_clearcoat) {
                f += _clearcoat->evaluate(wo_local, wi_local);
                pdf += _sampling_weights[_clearcoat_technique_index] *
                       _clearcoat->pdf(wo_local, wi_local);
            }
            cos_theta_i = ite(cls->it()->shape()->shadow_terminator_factor() > 0.f |
                                  cls->it()->same_sided(wo, wi),
                              abs_cos_theta(wi_local), 0.f);
        }
        $else {// transmission
            if (_spec_trans) {
                f += _spec_trans->evaluate(wo_local, wi_local, mode);
                pdf += _sampling_weights[_spec_trans_technique_index] *
                       _spec_trans->pdf(wo_local, wi_local, mode);
            }
            if (_diff_trans) {
                f += _diff_trans->evaluate(wo_local, wi_local, mode);
                pdf += _sampling_weights[_diff_trans_technique_index] *
                       _diff_trans->pdf(wo_local, wi_local, mode);
            }
            cos_theta_i = abs_cos_theta(wi_local);
        };
        return {.f = f * cos_theta_i, .pdf = pdf};
    }
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _color; }
    [[nodiscard]] Float2 roughness() const noexcept override {
        return DisneyMicrofacetDistribution::alpha_to_roughness(_distrib->alpha());
    }
    [[nodiscard]] Surface::Evaluation evaluate(const Surface::Closure *cls,
                                               Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode) const noexcept override {
        auto wo_local = cls->it()->shading().world_to_local(wo);
        auto wi_local = cls->it()->shading().world_to_local(wi);
        return _evaluate_local(cls, wo, wi, wo_local, wi_local, mode);
    }
    [[nodiscard]] Surface::Sample sample(const Surface::Closure *cls,
                                         Expr<float3> wo, Expr<float> u_lobe,
                                         Expr<float2> u, TransportMode mode) const noexcept override {
        auto sampling_tech = def(0u);
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            sampling_tech = ite(u_lobe > sum_weights, i, sampling_tech);
            sum_weights += _sampling_weights[i];
        }
        // sample
        auto wo_local = cls->it()->shading().world_to_local(wo);
        auto event = def(Surface::event_reflect);
        BxDF::SampledDirection wi_sample;
        $switch(sampling_tech) {
            if (_diffuse) {
                $case(_diffuse_like_technique_index) {
                    wi_sample = _diffuse->sample_wi(wo_local, u, mode);
                };
            }
            if (_specular) {
                $case(_specular_technique_index) {
                    wi_sample = _specular->sample_wi(wo_local, u, mode);
                };
            }
            if (_clearcoat) {
                $case(_clearcoat_technique_index) {
                    wi_sample = _clearcoat->sample_wi(wo_local, u);
                };
            }
            if (_spec_trans) {
                $case(_spec_trans_technique_index) {
                    wi_sample = _spec_trans->sample_wi(wo_local, u, mode);
                    event = Surface::event_through;
                };
            }
            if (_diff_trans) {
                $case(_diff_trans_technique_index) {
                    wi_sample = _diff_trans->sample_wi(wo_local, u, mode);
                    event = Surface::event_through;
                };
            }
            $default { unreachable(); };
        };
        auto eval = Surface::Evaluation::zero(_color.dimension());
        auto wi = cls->it()->shading().local_to_world(wi_sample.wi);
        $if(wi_sample.valid) {
            eval = _evaluate_local(cls, wo, wi, wo_local, wi_sample.wi, mode);
        };
        return {.eval = eval, .wi = wi, .event = event};
    }
};

class DisneySurfaceClosure : public Surface::Closure {

private:
    luisa::unique_ptr<DisneyClosureImplBase> _impl;

public:
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _impl->albedo(); }
    [[nodiscard]] Float2 roughness() const noexcept override { return _impl->roughness(); }

private:
    [[nodiscard]] optional<Float> _eta() const noexcept override { return _impl->eta(); }
    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        return _impl->evaluate(this, wo, wi, mode);
    }
    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo, Expr<float> u_lobe,
                                          Expr<float2> u, TransportMode mode) const noexcept override {
        return _impl->sample(this, wo, u_lobe, u, mode);
    }

public:
    DisneySurfaceClosure(const DisneySurfaceInstance *instance, luisa::shared_ptr<Interaction> it,
                         const SampledWavelengths &swl, Expr<float> eta_i, Expr<float> time,
                         const Texture::Instance *color_tex, const Texture::Instance *metallic_tex,
                         const Texture::Instance *eta_tex, const Texture::Instance *roughness_tex,
                         const Texture::Instance *spec_tint_tex, const Texture::Instance *aniso_tex,
                         const Texture::Instance *sheen_tex, const Texture::Instance *sheen_tint_tex,
                         const Texture::Instance *clearcoat_tex, const Texture::Instance *clearcoat_gloss_tex,
                         const Texture::Instance *spec_trans_tex, const Texture::Instance *flatness_tex,
                         const Texture::Instance *diff_trans_tex) noexcept
        : Surface::Closure{instance, std::move(it), swl, time} {
        if (instance->node()->is_thin()) {
            _impl = luisa::make_unique<ThinDisneySurfaceClosureImpl>(
                this, eta_i,
                color_tex, metallic_tex, eta_tex, roughness_tex, spec_tint_tex,
                aniso_tex, sheen_tex, sheen_tint_tex, clearcoat_tex,
                clearcoat_gloss_tex, spec_trans_tex, flatness_tex, diff_trans_tex);
        } else {
            _impl = luisa::make_unique<DisneySurfaceClosureImpl>(
                this, eta_i,
                color_tex, metallic_tex, eta_tex, roughness_tex, spec_tint_tex,
                aniso_tex, sheen_tex, sheen_tint_tex, clearcoat_tex,
                clearcoat_gloss_tex, spec_trans_tex, flatness_tex);
        }
    }
};

luisa::unique_ptr<Surface::Closure> DisneySurfaceInstance::closure(
    luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
    Expr<float> eta_i, Expr<float> time) const noexcept {
    return luisa::make_unique<DisneySurfaceClosure>(
        this, std::move(it), swl, eta_i, time,
        _color, _metallic, _eta, _roughness, _specular_tint,
        _anisotropic, _sheen, _sheen_tint, _clearcoat,
        _clearcoat_gloss, _specular_trans, _flatness, _diffuse_trans);
}

using NormalMapOpacityDisneySurface = NormalMapWrapper<OpacitySurfaceWrapper<
    DisneySurface, DisneySurfaceInstance, DisneySurfaceClosure>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapOpacityDisneySurface)

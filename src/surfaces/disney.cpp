//
// Created by Mike Smith on 2022/1/30.
//

#include <utility>
#include <dsl/builtin.h>
#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

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
    struct Gradient {
        SampledSpectrum dR;
    };

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
            return reflect(-wo, wh);
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
    [[nodiscard]] Gradient backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df) const noexcept {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
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
    [[nodiscard]] Float G(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        return G1(wo) * G1(wi);
    }
    [[nodiscard]] Gradient grad_G(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto d_alpha = grad_G1(wo).dAlpha * G1(wi) + G1(wo) * grad_G1(wi).dAlpha;
        return {.dAlpha = d_alpha};
    }
};

}// namespace

struct DisneyContext {
    Interaction it;
    SampledSpectrum color;
    Float color_lum;
    Float metallic;
    Float eta_i;
    Float eta_t;
    Float roughness;
    Float specular_tint;
    Float anisotropic;
    Float sheen;
    Float sheen_tint;
    Float clearcoat;
    Float clearcoat_gloss;
    Float specular_trans;
    Float flatness;
    Float diffuse_trans;
};

static constexpr auto disney_lobe_diffuse_bit = 1u << 0u;
static constexpr auto disney_lobe_retro_bit = 1u << 1u;
static constexpr auto disney_lobe_fake_ss_bit = 1u << 2u;
static constexpr auto disney_lobe_sheen_bit = 1u << 3u;
static constexpr auto disney_lobe_clearcoat_bit = 1u << 4u;
static constexpr auto disney_lobe_specular_bit = 1u << 5u;
static constexpr auto disney_lobe_diff_trans_bit = 1u << 6u;
static constexpr auto disney_lobe_spec_trans_bit = 1u << 7u;

class DisneyClosureImplBase {

public:
    virtual ~DisneyClosureImplBase() noexcept = default;

    [[nodiscard]] virtual SampledSpectrum albedo() const noexcept = 0;
    [[nodiscard]] virtual Float2 roughness() const noexcept = 0;
    [[nodiscard]] virtual luisa::optional<Float> eta() const noexcept = 0;

    [[nodiscard]] virtual Surface::Evaluation evaluate(
        Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept = 0;
    [[nodiscard]] virtual Surface::Sample sample(
        Expr<float3> wo, Expr<float> u_lobe,
        Expr<float2> u, TransportMode mode) const noexcept = 0;
};

class DisneyClosureImpl : public DisneyClosureImplBase {

public:
    static constexpr auto diffuse_like_technique_index = 0u;
    static constexpr auto specular_technique_index = 1u;
    static constexpr auto clearcoat_technique_index = 2u;
    static constexpr auto spec_trans_technique_index = 3u;
    static constexpr auto max_sampling_technique_count = 4u;

private:
    const DisneyContext &_ctx;
    luisa::unique_ptr<DisneyDiffuse> _diffuse;
    luisa::unique_ptr<DisneyFakeSS> _fake_ss;
    luisa::unique_ptr<DisneyRetro> _retro;
    luisa::unique_ptr<DisneySheen> _sheen;
    luisa::unique_ptr<DisneyMicrofacetDistribution> _distrib;
    luisa::unique_ptr<DisneyFresnel> _fresnel;
    luisa::unique_ptr<MicrofacetReflection> _specular;
    luisa::unique_ptr<DisneyClearcoat> _clearcoat;
    luisa::unique_ptr<MicrofacetTransmission> _spec_trans;
    uint _sampling_technique_count;
    std::array<Float, max_sampling_technique_count> _sampling_weights;
    std::bitset<max_sampling_technique_count> _enabled_sampling_techniques;

public:
    DisneyClosureImpl(const DisneyContext &ctx,
                      uint enabled_lobes,
                      bool is_transmissive) noexcept
        : _ctx{ctx},
          _sampling_technique_count{
              is_transmissive ?
                  max_sampling_technique_count :
                  max_sampling_technique_count - 1u} {

        auto diffuse_weight = (1.f - _ctx.metallic) * (1.f - _ctx.specular_trans);
        auto tint_weight = ite(_ctx.color_lum > 0.f, 1.f / _ctx.color_lum, 1.f);
        auto tint = saturate(_ctx.color * tint_weight);// normalize lum. to isolate hue+sat
        auto tint_lum = _ctx.color_lum * tint_weight;

        _enabled_sampling_techniques.reset();

        // diffuse-like lobes: diffuse, retro-reflection, and optionally fake subsurface scattering and sheen
        auto diffuse_like_sampling_weight = diffuse_weight * _ctx.color_lum;
        if ((enabled_lobes & disney_lobe_diffuse_bit) ||
            (enabled_lobes & disney_lobe_retro_bit)) {
            auto Cdiff_weight = diffuse_weight * (1.f - _ctx.flatness);
            auto Cdiff = _ctx.color * Cdiff_weight;
            _diffuse = luisa::make_unique<DisneyDiffuse>(Cdiff);
            _retro = luisa::make_unique<DisneyRetro>(Cdiff, _ctx.roughness);
            _enabled_sampling_techniques.set(diffuse_like_technique_index);
        }
        // fake subsurface scattering
        if (enabled_lobes & disney_lobe_fake_ss_bit) {
            auto Css_weight = diffuse_weight * _ctx.flatness;
            auto Css = Css_weight * _ctx.color;
            _fake_ss = luisa::make_unique<DisneyFakeSS>(Css, _ctx.roughness);
            _enabled_sampling_techniques.set(diffuse_like_technique_index);
        }
        // sheen
        if (enabled_lobes & disney_lobe_sheen_bit) {
            auto Csheen_weight = diffuse_weight * _ctx.sheen;
            auto Csheen = Csheen_weight * lerp(1.f, tint, _ctx.sheen_tint);
            _sheen = luisa::make_unique<DisneySheen>(Csheen);
            auto sheen_lum = Csheen_weight * lerp(1.f, tint_lum, _ctx.sheen_tint);
            diffuse_like_sampling_weight += sheen_lum * .1f;
            _enabled_sampling_techniques.set(diffuse_like_technique_index);
        }
        _sampling_weights[diffuse_like_technique_index] = saturate(diffuse_like_sampling_weight);

        // specular lobes: clearcoat, microfacet reflection, and optionally microfacet transmission
        auto eta = _ctx.eta_t / _ctx.eta_i;
        // specular is Trowbridge-Reitz with a modified Fresnel function
        auto SchlickR0 = SchlickR0FromEta(eta);
        auto Cspec0 = lerp(lerp(1.f, tint, _ctx.specular_tint) * SchlickR0, _ctx.color, _ctx.metallic);
        _fresnel = luisa::make_unique<DisneyFresnel>(Cspec0, _ctx.metallic, eta, !is_transmissive);

        // create the microfacet distribution for metallic and/or specular transmittance
        auto aspect = sqrt(1.f - _ctx.anisotropic * .9f);
        auto alpha = make_float2(max(0.001f, _ctx.roughness / aspect),
                                 max(0.001f, _ctx.roughness * aspect));
        _distrib = luisa::make_unique<DisneyMicrofacetDistribution>(alpha);
        _specular = luisa::make_unique<MicrofacetReflection>(
            SampledSpectrum{_ctx.color.dimension(), 1.f},
            _distrib.get(), _fresnel.get());

        // specular reflection sampling weight
        auto Cspec0_lum = lerp(lerp(1.f, tint_lum, _ctx.specular_tint) * SchlickR0,
                               _ctx.color_lum, _ctx.metallic);
        _sampling_weights[specular_technique_index] = saturate(Cspec0_lum);
        _enabled_sampling_techniques.set(specular_technique_index);

        // clearcoat
        if (enabled_lobes & disney_lobe_clearcoat_bit) {
            auto gloss = lerp(.1f, .001f, _ctx.clearcoat_gloss);
            _clearcoat = luisa::make_unique<DisneyClearcoat>(_ctx.clearcoat, gloss);
            // clearcoat sampling weight
            _sampling_weights[clearcoat_technique_index] = saturate(_ctx.clearcoat * FrSchlick(.04f, 1.f));
            _enabled_sampling_techniques.set(clearcoat_technique_index);
        }

        // specular transmission
        if (is_transmissive) {
            if (enabled_lobes & disney_lobe_spec_trans_bit) {
                auto Cst_weight = (1.f - _ctx.metallic) * _ctx.specular_trans;
                auto Cst = Cst_weight * sqrt(_ctx.color);
                _spec_trans = luisa::make_unique<MicrofacetTransmission>(
                    Cst, _distrib.get(), _ctx.eta_i, _ctx.eta_t);
                auto Cst_lum = Cst_weight * sqrt(_ctx.color_lum);
                // specular transmission sampling weight
                _sampling_weights[spec_trans_technique_index] = saturate(Cst_lum);
                _enabled_sampling_techniques.set(spec_trans_technique_index);
            }
        }

        // normalize sampling weights
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            if (_enabled_sampling_techniques.test(i)) {
                sum_weights += _sampling_weights[i];
            }
        }
        auto inv_sum_weights = ite(sum_weights == 0.f, 0.f, 1.f / sum_weights);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            if (_enabled_sampling_techniques.test(i)) {
                _sampling_weights[i] *= inv_sum_weights;
            }
        }
    }

private:
    [[nodiscard]] Surface::Evaluation _evaluate_local(Expr<float3> wo_local,
                                                      Expr<float3> wi_local,
                                                      TransportMode mode) const noexcept {
        SampledSpectrum f{_ctx.color.dimension(), 0.f};
        auto pdf = def(0.f);
        $outline {
            $if(same_hemisphere(wo_local, wi_local)) {// reflection
                if (_diffuse) {
                    $if(_sampling_weights[diffuse_like_technique_index] > 0.f) {
                        f += _diffuse->evaluate(wo_local, wi_local, mode);
                        f += _retro->evaluate(wo_local, wi_local, mode);
                        if (_fake_ss) { f += _fake_ss->evaluate(wo_local, wi_local, mode); }
                        if (_sheen) { f += _sheen->evaluate(wo_local, wi_local, mode); }
                        pdf += _sampling_weights[diffuse_like_technique_index] *
                               _diffuse->pdf(wo_local, wi_local, mode);
                    };
                }
                if (_specular) {
                    $if(_sampling_weights[specular_technique_index] > 0.f) {
                        f += _specular->evaluate(wo_local, wi_local, mode);
                        pdf += _sampling_weights[specular_technique_index] *
                               _specular->pdf(wo_local, wi_local, mode);
                    };
                }
                if (_clearcoat) {
                    $if(_sampling_weights[clearcoat_technique_index] > 0.f) {
                        f += _clearcoat->evaluate(wo_local, wi_local);
                        pdf += _sampling_weights[clearcoat_technique_index] *
                               _clearcoat->pdf(wo_local, wi_local);
                    };
                }
            }
            $else {// transmission
                if (_spec_trans) {
                    $if(_sampling_weights[spec_trans_technique_index] > 0.f) {
                        f += _spec_trans->evaluate(wo_local, wi_local, mode);
                        pdf += _sampling_weights[spec_trans_technique_index] *
                               _spec_trans->pdf(wo_local, wi_local, mode);
                    };
                }
            };
        };
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }

public:
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _ctx.color; }
    [[nodiscard]] Float2 roughness() const noexcept override {
        return DisneyMicrofacetDistribution::alpha_to_roughness(_distrib->alpha());
    }
    [[nodiscard]] luisa::optional<Float> eta() const noexcept override {
        return _spec_trans ? luisa::make_optional(_ctx.eta_t) : luisa::nullopt;
    }

    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode) const noexcept override {
        auto wo_local = _ctx.it.shading().world_to_local(wo);
        auto wi_local = _ctx.it.shading().world_to_local(wi);
        return _evaluate_local(wo_local, wi_local, mode);
    }

    [[nodiscard]] Surface::Sample sample(Expr<float3> wo, Expr<float> u_lobe,
                                         Expr<float2> u, TransportMode mode) const noexcept override {
        auto sampling_tech = def(0u);
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < _sampling_technique_count; i++) {
            if (_enabled_sampling_techniques.test(i)) {
                sampling_tech = ite(u_lobe > sum_weights, i, sampling_tech);
                sum_weights += _sampling_weights[i];
            }
        }
        // sample
        auto wo_local = _ctx.it.shading().world_to_local(wo);
        auto event = def(Surface::event_reflect);
        BxDF::SampledDirection wi_sample{.valid = false};
        $outline {
            $switch(sampling_tech) {
                if (_diffuse) {
                    $case(diffuse_like_technique_index) {
                        wi_sample = _diffuse->sample_wi(wo_local, u, mode);
                    };
                }
                if (_specular) {
                    $case(specular_technique_index) {
                        wi_sample = _specular->sample_wi(wo_local, u, mode);
                    };
                }
                if (_clearcoat) {
                    $case(clearcoat_technique_index) {
                        wi_sample = _clearcoat->sample_wi(wo_local, u);
                    };
                }
                if (_spec_trans) {
                    $case(spec_trans_technique_index) {
                        wi_sample = _spec_trans->sample_wi(wo_local, u, mode);
                        event = ite(cos_theta(wo_local) > 0.f, Surface::event_enter, Surface::event_exit);
                    };
                }
            };
        };
        auto eval = Surface::Evaluation::zero(_ctx.color.dimension());
        auto wi = _ctx.it.shading().local_to_world(wi_sample.wi);
        $if(wi_sample.valid) {
            eval = _evaluate_local(wo_local, wi_sample.wi, mode);
        };
        return {.eval = eval, .wi = wi, .event = event};
    }
};

class ThinDisneyClosureImpl : public DisneyClosureImplBase {

public:
    static constexpr auto diffuse_like_technique_index = 0u;
    static constexpr auto specular_technique_index = 1u;
    static constexpr auto clearcoat_technique_index = 2u;
    static constexpr auto spec_trans_technique_index = 3u;
    static constexpr auto diff_trans_technique_index = 4u;
    static constexpr auto sampling_technique_count = 5u;

private:
    const DisneyContext &_ctx;
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
    std::array<Float, sampling_technique_count> _sampling_weights;
    std::bitset<sampling_technique_count> _enabled_sampling_techniques;

public:
    ThinDisneyClosureImpl(const DisneyContext &ctx,
                          uint enabled_lobes) noexcept : _ctx{ctx} {

        auto diffuse_weight = (1.f - _ctx.metallic) * (1.f - _ctx.specular_trans);
        auto diff_refl_weight = diffuse_weight * (1.f - _ctx.diffuse_trans);
        auto diff_trans_weight = diffuse_weight * _ctx.diffuse_trans;
        auto tint_weight = ite(_ctx.color_lum > 0.f, 1.f / _ctx.color_lum, 1.f);
        auto tint = saturate(_ctx.color * tint_weight);// normalize lum. to isolate hue+sat
        auto tint_lum = _ctx.color_lum * tint_weight;

        _enabled_sampling_techniques.reset();

        // diffuse-like lobes: diffuse, retro-reflection, fake subsurface scattering and sheen
        auto diffuse_like_sampling_weight = diff_refl_weight * _ctx.color_lum;
        if ((enabled_lobes & disney_lobe_diffuse_bit) ||
            (enabled_lobes & disney_lobe_retro_bit)) {
            auto Cdiff_weight = diff_refl_weight * (1.f - _ctx.flatness);
            auto Cdiff = _ctx.color * Cdiff_weight;
            _diffuse = luisa::make_unique<DisneyDiffuse>(Cdiff);
            _retro = luisa::make_unique<DisneyRetro>(Cdiff, _ctx.roughness);
            _enabled_sampling_techniques.set(diffuse_like_technique_index);
        }

        if (enabled_lobes & disney_lobe_fake_ss_bit) {
            auto Css_weight = diff_refl_weight * _ctx.flatness * (1.f - _ctx.diffuse_trans);
            auto Css = Css_weight * _ctx.color;
            _fake_ss = luisa::make_unique<DisneyFakeSS>(Css, _ctx.roughness);
            _enabled_sampling_techniques.set(diffuse_like_technique_index);
        }

        // sheen
        if (enabled_lobes & disney_lobe_sheen_bit) {
            auto Csheen_weight = diff_refl_weight * _ctx.sheen * (1.f - _ctx.diffuse_trans);
            auto Csheen = Csheen_weight * lerp(1.f, tint, _ctx.sheen_tint);
            _sheen = luisa::make_unique<DisneySheen>(Csheen);
            auto sheen_lum = Csheen_weight * lerp(1.f, tint_lum, _ctx.sheen_tint);
            diffuse_like_sampling_weight += sheen_lum * .1f;
        }
        _sampling_weights[diffuse_like_technique_index] = saturate(diffuse_like_sampling_weight);

        // specular lobes: clearcoat, microfacet reflection, and optionally microfacet transmission
        auto eta = _ctx.eta_t / _ctx.eta_i;
        // specular is Trowbridge-Reitz with a modified Fresnel function
        auto SchlickR0 = SchlickR0FromEta(eta);
        auto Cspec0 = lerp(lerp(1.f, tint, _ctx.specular_tint) * SchlickR0, _ctx.color, _ctx.metallic);
        _fresnel = luisa::make_unique<DisneyFresnel>(Cspec0, _ctx.metallic, eta, false);

        // create the microfacet distribution for metallic and/or specular transmittance
        auto aspect = sqrt(1.f - _ctx.anisotropic * .9f);
        auto alpha = make_float2(max(0.001f, _ctx.roughness / aspect),
                                 max(0.001f, _ctx.roughness * aspect));
        _distrib = luisa::make_unique<DisneyMicrofacetDistribution>(alpha);
        _specular = luisa::make_unique<MicrofacetReflection>(
            SampledSpectrum{_ctx.color.dimension(), 1.f}, _distrib.get(), _fresnel.get());

        // specular reflection sampling weight
        auto Cspec0_lum = lerp(lerp(1.f, tint_lum, _ctx.specular_tint) * SchlickR0, _ctx.color_lum, _ctx.metallic);
        _sampling_weights[specular_technique_index] = saturate(Cspec0_lum);
        _enabled_sampling_techniques.set(specular_technique_index);

        // clearcoat
        if (enabled_lobes & disney_lobe_clearcoat_bit) {
            auto gloss = lerp(.1f, .001f, ctx.clearcoat_gloss);
            _clearcoat = luisa::make_unique<DisneyClearcoat>(_ctx.clearcoat, gloss);
            // clearcoat sampling weight
            _sampling_weights[clearcoat_technique_index] = saturate(_ctx.clearcoat * FrSchlick(.04f, 1.f));
            _enabled_sampling_techniques.set(clearcoat_technique_index);
        }

        // thin specular transmission distribution
        if (enabled_lobes & disney_lobe_spec_trans_bit) {
            auto rscaled = (.65f * eta - .35f) * _ctx.roughness;
            auto ascaled = make_float2(max(.001f, rscaled / aspect),
                                       max(.001f, rscaled * aspect));
            _thin_distrib = luisa::make_unique<TrowbridgeReitzDistribution>(ascaled);
            auto Cst_weight = (1.f - _ctx.metallic) * _ctx.specular_trans;
            auto Cst = Cst_weight * _ctx.color;
            _spec_trans = luisa::make_unique<MicrofacetTransmission>(
                Cst, _thin_distrib.get(), _ctx.eta_i, _ctx.eta_t);
            auto Cst_lum = Cst_weight * _ctx.color_lum;
            // specular transmission sampling weight
            _sampling_weights[spec_trans_technique_index] = saturate(Cst_lum);
            _enabled_sampling_techniques.set(spec_trans_technique_index);
        }

        // diffuse transmission
        if (enabled_lobes & disney_lobe_diff_trans_bit) {
            auto Cdt = diff_trans_weight * _ctx.color;
            auto Cdt_lum = diff_trans_weight * _ctx.color_lum;
            _diff_trans = luisa::make_unique<LambertianTransmission>(Cdt);
            _sampling_weights[diff_trans_technique_index] = saturate(Cdt_lum);
            _enabled_sampling_techniques.set(diff_trans_technique_index);
        }

        // normalize sampling weights
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < sampling_technique_count; i++) {
            if (_enabled_sampling_techniques.test(i)) {
                sum_weights += _sampling_weights[i];
            }
        }
        auto inv_sum_weights = ite(sum_weights == 0.f, 0.f, 1.f / sum_weights);
        for (auto i = 0u; i < sampling_technique_count; i++) {
            if (_enabled_sampling_techniques.test(i)) {
                _sampling_weights[i] *= inv_sum_weights;
            }
        }
    }

private:
    [[nodiscard]] Surface::Evaluation _evaluate_local(Expr<float3> wo_local,
                                                      Expr<float3> wi_local,
                                                      TransportMode mode) const noexcept {
        SampledSpectrum f{_ctx.color.dimension(), 0.f};
        auto pdf = def(0.f);
        $if(same_hemisphere(wo_local, wi_local)) {// reflection
            if (_diffuse) {
                $if(_sampling_weights[diffuse_like_technique_index] > 0.f) {
                    f += _diffuse->evaluate(wo_local, wi_local, mode);
                    f += _retro->evaluate(wo_local, wi_local, mode);
                    if (_fake_ss) { f += _fake_ss->evaluate(wo_local, wi_local, mode); }
                    if (_sheen) { f += _sheen->evaluate(wo_local, wi_local, mode); }
                    pdf += _sampling_weights[diffuse_like_technique_index] *
                           _diffuse->pdf(wo_local, wi_local, mode);
                };
            }
            if (_specular) {
                $if(_sampling_weights[specular_technique_index] > 0.f) {
                    f += _specular->evaluate(wo_local, wi_local, mode);
                    pdf += _sampling_weights[specular_technique_index] *
                           _specular->pdf(wo_local, wi_local, mode);
                };
            }
            if (_clearcoat) {
                $if(_sampling_weights[clearcoat_technique_index] > 0.f) {
                    f += _clearcoat->evaluate(wo_local, wi_local);
                    pdf += _sampling_weights[clearcoat_technique_index] *
                           _clearcoat->pdf(wo_local, wi_local);
                };
            }
        }
        $else {// transmission
            if (_spec_trans) {
                $if(_sampling_weights[spec_trans_technique_index] > 0.f) {
                    f += _spec_trans->evaluate(wo_local, wi_local, mode);
                    pdf += _sampling_weights[spec_trans_technique_index] *
                           _spec_trans->pdf(wo_local, wi_local, mode);
                };
            }
            if (_diff_trans) {
                $if(_sampling_weights[diff_trans_technique_index] > 0.f) {
                    f += _diff_trans->evaluate(wo_local, wi_local, mode);
                    pdf += _sampling_weights[diff_trans_technique_index] *
                           _diff_trans->pdf(wo_local, wi_local, mode);
                };
            }
        };
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }

public:
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _ctx.color; }
    [[nodiscard]] luisa::optional<Float> eta() const noexcept override { return luisa::nullopt; }
    [[nodiscard]] Float2 roughness() const noexcept override {
        return DisneyMicrofacetDistribution::alpha_to_roughness(_distrib->alpha());
    }

    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        auto wo_local = _ctx.it.shading().world_to_local(wo);
        auto wi_local = _ctx.it.shading().world_to_local(wi);
        return _evaluate_local(wo_local, wi_local, mode);
    }

    [[nodiscard]] Surface::Sample sample(Expr<float3> wo,
                                         Expr<float> u_lobe, Expr<float2> u,
                                         TransportMode mode) const noexcept override {
        auto sampling_tech = def(0u);
        auto sum_weights = def(0.f);
        for (auto i = 0u; i < sampling_technique_count; i++) {
            if (_enabled_sampling_techniques.test(i)) {
                sampling_tech = ite(u_lobe > sum_weights, i, sampling_tech);
                sum_weights += _sampling_weights[i];
            }
        }
        // sample
        auto wo_local = _ctx.it.shading().world_to_local(wo);
        auto event = def(Surface::event_reflect);
        BxDF::SampledDirection wi_sample{.valid = false};
        $switch(sampling_tech) {
            if (_diffuse) {
                $case(diffuse_like_technique_index) {
                    wi_sample = _diffuse->sample_wi(wo_local, u, mode);
                };
            }
            if (_specular) {
                $case(specular_technique_index) {
                    wi_sample = _specular->sample_wi(wo_local, u, mode);
                };
            }
            if (_clearcoat) {
                $case(clearcoat_technique_index) {
                    wi_sample = _clearcoat->sample_wi(wo_local, u);
                };
            }
            if (_spec_trans) {
                $case(spec_trans_technique_index) {
                    wi_sample = _spec_trans->sample_wi(wo_local, u, mode);
                    event = Surface::event_through;
                };
            }
            if (_diff_trans) {
                $case(diff_trans_technique_index) {
                    wi_sample = _diff_trans->sample_wi(wo_local, u, mode);
                    event = Surface::event_through;
                };
            }
        };
        auto eval = Surface::Evaluation::zero(_ctx.color.dimension());
        auto wi = _ctx.it.shading().local_to_world(wi_sample.wi);
        $if(wi_sample.valid) {
            eval = _evaluate_local(wo_local, wi_sample.wi, mode);
        };
        return {.eval = eval, .wi = wi, .event = event};
    }
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

private:
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

public:
    [[nodiscard]] luisa::string closure_identifier() const noexcept override {
        auto s = node<DisneySurface>();
        return s->is_thin()         ? "disney_thin" :
               s->is_transmissive() ? "disney_trans" :
                                      "disney";
    }

    void populate_closure(Surface::Closure *closure, const Interaction &it,
                          Expr<float3> wo, Expr<float> eta_i) const noexcept override;

    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

class DisneySurfaceClosure : public Surface::Closure {

private:
    bool _thin;
    bool _transmissive;
    uint _enabled_lobes = 0u;
    luisa::unique_ptr<DisneyClosureImplBase> _impl;

public:
    DisneySurfaceClosure(const DisneySurfaceInstance *instance,
                         const Pipeline &pipeline,
                         const SampledWavelengths &swl,
                         Expr<float> time,
                         bool is_thin, bool is_transmissive) noexcept
        : Surface::Closure{instance, pipeline, swl, time},
          _thin{is_thin}, _transmissive{is_transmissive} {}

public:
    void pre_eval() noexcept override {
        if (_thin) {
            _impl = luisa::make_unique<ThinDisneyClosureImpl>(
                context<DisneyContext>(), _enabled_lobes);
        } else {
            _impl = luisa::make_unique<DisneyClosureImpl>(
                context<DisneyContext>(), _enabled_lobes, _transmissive);
        }
    }
    void post_eval() noexcept override { _impl = nullptr; }
    void enable_lobes(uint lobe_mask) noexcept { _enabled_lobes |= lobe_mask; }
    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return _impl->albedo(); }
    [[nodiscard]] Float2 roughness() const noexcept override { return _impl->roughness(); }
    [[nodiscard]] luisa::optional<Float> eta() const noexcept override { return _impl->eta(); }
    [[nodiscard]] const Interaction &it() const noexcept override { return context<DisneyContext>().it; }

private:
    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        return _impl->evaluate(wo, wi, mode);
    }
    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo, Expr<float> u_lobe,
                                          Expr<float2> u, TransportMode mode) const noexcept override {
        return _impl->sample(wo, u_lobe, u, mode);
    }
    void _backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df,
                   TransportMode mode) const noexcept override {
        // TODO
        LUISA_WARNING_WITH_LOCATION("Not implemented.");
    }
};

void DisneySurfaceInstance::populate_closure(Surface::Closure *closure, const Interaction &it,
                                             Expr<float3> wo, Expr<float> eta_i) const noexcept {
    auto &swl = closure->swl();
    auto time = closure->time();
    auto color_decode = _color ? _color->evaluate_albedo_spectrum(it, swl, time) :
                                 Spectrum::Decode::one(swl.dimension());
    auto metallic = _metallic ? _metallic->evaluate(it, swl, time).x : 0.f;
    auto eta = _eta ? _eta->evaluate(it, swl, time).x : 1.5f;
    auto roughness = _roughness ? _roughness->evaluate(it, swl, time).x : .5f;
    if (node<DisneySurface>()->remap_roughness()) {
        roughness = DisneyMicrofacetDistribution::roughness_to_alpha(roughness);
    }
    auto specular_tint = _specular_tint ? _specular_tint->evaluate(it, swl, time).x : 0.f;
    auto anisotropic = _anisotropic ? _anisotropic->evaluate(it, swl, time).x : 0.f;
    auto sheen = _sheen ? _sheen->evaluate(it, swl, time).x : 0.f;
    auto sheen_tint = _sheen_tint ? _sheen_tint->evaluate(it, swl, time).x : 0.f;
    auto clearcoat = _clearcoat ? _clearcoat->evaluate(it, swl, time).x : 0.f;
    auto clearcoat_gloss = _clearcoat_gloss ? _clearcoat_gloss->evaluate(it, swl, time).x : 1.f;
    auto specular_trans = _specular_trans ? _specular_trans->evaluate(it, swl, time).x : 0.f;
    auto flatness = _flatness ? _flatness->evaluate(it, swl, time).x : 0.f;
    auto diffuse_trans = _diffuse_trans ? _diffuse_trans->evaluate(it, swl, time).x : 0.f;

    DisneyContext ctx{
        .it = it,
        .color = color_decode.value,
        .color_lum = color_decode.strength,
        .metallic = metallic,
        .eta_i = eta_i,
        .eta_t = eta,
        .roughness = roughness,
        .specular_tint = specular_tint,
        .anisotropic = anisotropic,
        .sheen = sheen,
        .sheen_tint = sheen_tint,
        .clearcoat = clearcoat,
        .clearcoat_gloss = clearcoat_gloss,
        .specular_trans = specular_trans,
        .flatness = flatness,
        .diffuse_trans = diffuse_trans};

    // find used lobes
    auto lobes = 0u;
    if (!_color || !_color->node()->is_black()) {
        lobes |= disney_lobe_diffuse_bit;
        lobes |= disney_lobe_retro_bit;
        if (_sheen && !_sheen->node()->is_black()) {
            lobes |= disney_lobe_sheen_bit;
        }
        if (_flatness && !_flatness->node()->is_black()) {
            lobes |= disney_lobe_fake_ss_bit;
        }
    }
    lobes |= disney_lobe_specular_bit;
    if (_clearcoat && !_clearcoat->node()->is_black()) {
        lobes |= disney_lobe_clearcoat_bit;
    }
    if (_specular_trans && !_specular_trans->node()->is_black()) {
        lobes |= disney_lobe_spec_trans_bit;
    }
    if (_diffuse_trans && !_diffuse_trans->node()->is_black()) {
        lobes |= disney_lobe_diff_trans_bit;
    }

    // update closure
    auto disney_closure = dynamic_cast<DisneySurfaceClosure *>(closure);
    disney_closure->bind(std::move(ctx));
    disney_closure->enable_lobes(lobes);
}

[[nodiscard]] luisa::unique_ptr<Surface::Closure> DisneySurfaceInstance::create_closure(
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return luisa::make_unique<DisneySurfaceClosure>(
        this,
        pipeline(), swl, time,
        node<DisneySurface>()->is_thin(),
        node<DisneySurface>()->is_transmissive());
}

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

using NormalMapOpacityDisneySurface = NormalMapWrapper<OpacitySurfaceWrapper<
    DisneySurface, DisneySurfaceInstance>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapOpacityDisneySurface)

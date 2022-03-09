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

public:
    struct Params {
        TextureHandle color;
        TextureHandle metallic;
        TextureHandle eta;
        TextureHandle roughness;
        TextureHandle specular_tint;
        TextureHandle anisotropic;
        TextureHandle sheen;
        TextureHandle sheen_tint;
        TextureHandle clearcoat;
        TextureHandle clearcoat_gloss;
        TextureHandle speculat_trans;
        TextureHandle flatness;
        TextureHandle diffuse_trans;
        bool thin;
    };

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
    [[nodiscard]] uint encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint instance_id, const Shape *shape) const noexcept override {
        Params params{
            .color = *pipeline.encode_texture(command_buffer, _color),
            .metallic = *pipeline.encode_texture(command_buffer, _metallic),
            .eta = *pipeline.encode_texture(command_buffer, _eta),
            .roughness = *pipeline.encode_texture(command_buffer, _roughness),
            .specular_tint = *pipeline.encode_texture(command_buffer, _specular_tint),
            .anisotropic = *pipeline.encode_texture(command_buffer, _anisotropic),
            .sheen = *pipeline.encode_texture(command_buffer, _sheen),
            .sheen_tint = *pipeline.encode_texture(command_buffer, _sheen_tint),
            .clearcoat = *pipeline.encode_texture(command_buffer, _clearcoat),
            .clearcoat_gloss = *pipeline.encode_texture(command_buffer, _clearcoat_gloss),
            .speculat_trans = *pipeline.encode_texture(command_buffer, _specular_trans),
            .flatness = *pipeline.encode_texture(command_buffer, _flatness),
            .diffuse_trans = *pipeline.encode_texture(command_buffer, _diffuse_trans),
            .thin = _thin};
        auto [buffer, buffer_id] = pipeline.arena_buffer<Params>(1u);
        command_buffer << buffer.copy_from(&params) << compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

}// namespace luisa::render

LUISA_STRUCT(
    luisa::render::DisneySurface::Params,
    color, metallic, eta, roughness, specular_tint, anisotropic,
    sheen, sheen_tint, clearcoat, clearcoat_gloss, speculat_trans,
    flatness, diffuse_trans, thin){};

namespace luisa::render {

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

[[nodiscard]] inline Float4 FrSchlick(Expr<float4> R0, Expr<float> cosTheta) noexcept {
    return lerp(R0, make_float4(1.f), SchlickWeight(cosTheta));
}

// For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
// coming from air..
[[nodiscard]] inline Float SchlickR0FromEta(Float eta) {
    return sqr(eta - 1.f) / sqr(eta + 1.f);
}

class DisneyDiffuse final : public BxDF {

private:
    Float4 R;

public:
    explicit DisneyDiffuse(Float4 R) noexcept : R{std::move(R)} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto Fo = SchlickWeight(abs_cos_theta(wo));
        auto Fi = SchlickWeight(abs_cos_theta(wi));

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
        // Burley 2015, eq (4).
        return R * inv_pi * (1.f - Fo * .5f) * (1.f - Fi * .5f);
    }
    [[nodiscard]] luisa::vector<Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto Fo = SchlickWeight(abs_cos_theta(wo));
        auto Fi = SchlickWeight(abs_cos_theta(wi));

        auto d_r = make_float4(inv_pi * (1.f - Fo * .5f) * (1.f - Fi * .5f));
        return {d_r};
    }
};

// "Fake" subsurface scattering lobe, based on the Hanrahan-Krueger BRDF
// approximation of the BSSRDF.
class DisneyFakeSS final : public BxDF {

private:
    Float4 R;
    Float roughness;

public:
    DisneyFakeSS(Float4 R, Float roughness) noexcept
        : R{std::move(R)}, roughness{std::move(roughness)} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
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
        return ite(valid, R * inv_pi * ss, 0.f);
    }
    [[nodiscard]] luisa::vector<Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
};

class DisneyRetro final : public BxDF {

private:
    Float4 R;
    Float roughness;

public:
    DisneyRetro(Float4 R, Float roughness) noexcept
        : R{std::move(R)}, roughness{std::move(roughness)} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto wh = wi + wo;
        auto valid = any(wh != 0.f);
        wh = normalize(wh);
        auto cosThetaD = dot(wi, wh);
        auto Fo = SchlickWeight(abs_cos_theta(wo));
        auto Fi = SchlickWeight(abs_cos_theta(wi));
        auto Rr = 2.f * roughness * cosThetaD * cosThetaD;

        // Burley 2015, eq (4).
        auto f = R * inv_pi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1.f));
        return ite(valid, f, 0.f);
    }
    [[nodiscard]] luisa::vector<Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
};

class DisneySheen final : public BxDF {

private:
    Float4 R;

public:
    explicit DisneySheen(Float4 R) noexcept : R{std::move(R)} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        auto wh = wi + wo;
        auto valid = any(wh != 0.f);
        wh = normalize(wh);
        auto cosThetaD = dot(wi, wh);
        return ite(valid, R * SchlickWeight(cosThetaD), 0.f);
    }
    [[nodiscard]] luisa::vector<Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
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

class DisneyClearcoat final : public BxDF {

private:
    Float weight;
    Float gloss;

public:
    DisneyClearcoat(Float weight, Float gloss) noexcept
        : weight{std::move(weight)}, gloss{std::move(gloss)} {}
    [[nodiscard]] Float4 evaluate(Expr<float3> wo, Expr<float3> wi) const noexcept override {
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
        return ite(valid, weight * Gr * Fr * Dr * .25f, make_float4(0.f));
    }
    [[nodiscard]] Float4 sample(Expr<float3> wo, Float3 *wi, Expr<float2> u, Float *p) const noexcept override {
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
    [[nodiscard]] Float pdf(Expr<float3> wo, Expr<float3> wi) const noexcept override {
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
    [[nodiscard]] luisa::vector<Float4> grad(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
};

// Specialized Fresnel function used for the specular component, based on
// a mixture between dielectric and the Schlick Fresnel approximation.
class DisneyFresnel final : public Fresnel {

private:
    Float4 R0;
    Float metallic;
    Float e;

public:
    DisneyFresnel(Float4 R0, Float metallic, Float eta) noexcept
        : R0{std::move(R0)}, metallic{std::move(metallic)}, e{std::move(eta)} {}
    [[nodiscard]] Float4 evaluate(Expr<float> cosI) const noexcept override {
        return lerp(
            fresnel_dielectric(cosI, make_float4(1.f), make_float4(e)),
            FrSchlick(R0, cosI),
            metallic);
    }
    [[nodiscard]] auto eta() const noexcept { return e; }
};

struct DisneyMicrofacetDistribution final : public TrowbridgeReitzDistribution {
    explicit DisneyMicrofacetDistribution(Float2 alpha) noexcept
        : TrowbridgeReitzDistribution{std::move(alpha)} {}
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
    const Interaction &_it;
    const SampledWavelengths &_swl;
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
    Bool _thin;
    Float _sampling_weights[max_sampling_techique_count];

public:
    DisneySurfaceClosure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> color, Expr<float> metallic_in, Expr<float> eta_in, Expr<float> roughness,
        Expr<float> specular_tint, Expr<float> anisotropic, Expr<float> sheen, Expr<float> sheen_tint,
        Expr<float> clearcoat, Expr<float> clearcoat_gloss, Expr<float> specular_trans_in,
        Expr<float> flatness, Expr<float> diffuse_trans, Expr<bool> thin) noexcept
        : _it{it}, _swl{swl}, _lobes{0u}, _thin{thin} {

        constexpr auto black_threshold = 1e-6f;
        auto cos_theta_o = dot(_it.wo(), _it.shading().n());
        auto front_face = cos_theta_o > 0.f;
        auto metallic = ite(front_face, metallic_in, 0.f);
        auto specular_trans = (1.f - metallic) * specular_trans_in;
        auto diffuse_weight = (1.f - metallic) * (1.f - specular_trans);
        auto dt = diffuse_trans * .5f;// 0: all diffuse is reflected -> 1, transmitted
        auto lum = swl.cie_y(color);
        auto Ctint = ite(lum > 0.f, color / lum, 1.f);// normalize lum. to isolate hue+sat
        auto eta = ite(eta_in < black_threshold, 1.5f, eta_in);

        // diffuse
        auto diffuse_scale = ite(thin, (1.f - flatness) * (1.f - dt), ite(front_face, 1.f, 0.f));
        auto Cdiff = diffuse_weight * color * diffuse_scale;
        _diffuse = luisa::make_unique<DisneyDiffuse>(Cdiff);
        _lobes |= ite(any(Cdiff > black_threshold), refl_diffuse, 0u);
        auto Css = ite(thin, diffuse_weight * flatness * (1.f - dt) * color, 0.f);
        _fake_ss = luisa::make_unique<DisneyFakeSS>(Css, roughness);
        _lobes |= ite(any(Css > black_threshold), refl_fake_ss, 0u);

        // retro-reflection
        auto Cretro = ite(front_face | thin, diffuse_weight * color, 0.f);
        _retro = luisa::make_unique<DisneyRetro>(Cretro, roughness);
        _lobes |= ite(any(Cretro > black_threshold), refl_retro, 0u);

        // sheen
        auto Csheen = ite(front_face | thin, diffuse_weight * sheen * lerp(specular_tint, 1.f, Ctint), 0.f);
        _sheen = luisa::make_unique<DisneySheen>(Csheen);
        _lobes |= ite(any(Csheen > black_threshold), refl_sheen, 0u);

        // diffuse sampling weight
        _sampling_weights[sampling_technique_diffuse] = swl.cie_y(Cdiff + Css + Cretro + Csheen);

        // create the microfacet distribution for metallic and/or specular transmittance
        auto aspect = sqrt(1.f - anisotropic * .9f);
        auto alpha = make_float2(
            max(0.001f, sqr(roughness) / aspect),
            max(0.001f, sqr(roughness) * aspect));
        _distrib = luisa::make_unique<DisneyMicrofacetDistribution>(alpha);

        // specular is Trowbridge-Reitz with a modified Fresnel function
        auto Cspec = lerp(1.f, Ctint, specular_tint);
        auto Cspec0 = lerp(SchlickR0FromEta(eta) * Cspec, color, metallic);
        _fresnel = luisa::make_unique<DisneyFresnel>(Cspec0, metallic, eta);
        _specular = luisa::make_unique<MicrofacetReflection>(
            make_float4(1.f), _distrib.get(), _fresnel.get());
        _lobes |= refl_specular;// always consider the specular lobe

        // specular reflection sampling weight
        auto F = saturate(_fresnel->evaluate(cos_theta_o).x);
        _sampling_weights[sampling_technique_specular] = swl.cie_y(Cspec0) * F;

        // clearcoat
        auto cc = ite(front_face | thin, clearcoat, 0.f);
        auto gloss = lerp(.1f, .001f, clearcoat_gloss);
        _clearcoat = luisa::make_unique<DisneyClearcoat>(cc, gloss);
        _lobes |= ite(cc > black_threshold, refl_clearcoat, 0u);

        // clearcoat sampling weight
        _sampling_weights[sampling_technique_clearcoat] = cc * FrSchlick(.04f, 1.f);

        // specular transmission
        auto T = specular_trans * sqrt(color);
        auto Cst = ite(thin, 0.f, T);
        _spec_trans = luisa::make_unique<MicrofacetTransmission>(
            Cst, _distrib.get(), make_float4(1.f), make_float4(eta));
        _lobes |= ite(any(Cst > black_threshold), trans_specular, 0u);

        // specular transmission sampling weight
        _sampling_weights[sampling_technique_specular_trans] = (1.f - F) * swl.cie_y(Cst);

        // thin specular transmission
        auto rscaled = (.65f * eta - .35f) * roughness;
        auto ascaled = make_float2(
            max(.001f, sqr(rscaled) / aspect),
            max(.001f, sqr(rscaled) * aspect));
        auto Ctst = ite(thin, T, 0.f);
        _thin_distrib = luisa::make_unique<TrowbridgeReitzDistribution>(ascaled);
        _thin_spec_trans = luisa::make_unique<MicrofacetTransmission>(
            Ctst, _thin_distrib.get(), make_float4(1.f), make_float4(eta));
        _lobes |= ite(any(Ctst > black_threshold), trans_thin_specular, 0u);

        // thin specular transmission sampling weight
        _sampling_weights[sampling_technique_thin_specular_trans] = (1.f - F) * swl.cie_y(Ctst);

        // thin diffuse transmission
        auto Cdt = ite(thin, dt * color, 0.f);
        _diff_trans = luisa::make_unique<LambertianTransmission>(dt * color);
        _lobes |= ite(any(Cdt > black_threshold), trans_thin_diffuse, 0u);

        // thin diffuse transmission sampling weight
        _sampling_weights[sampling_technique_thin_diffuse_trans] = swl.cie_y(Cdt);

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
        auto f = def(make_float4());
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
        return {.swl = _swl, .f = f, .pdf = pdf, .alpha = _distrib->alpha(),
                .eta = ite(_thin & wi_local.z < 0.f, make_float4(_fresnel->eta()), 1.f)};
    }
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        return evaluate_local(wo_local, wi_local);
    }
    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {

        // TODO: weighted sampling

        auto u = sampler.generate_2d();
        auto sampling_tech = def(0u);
        auto sum_weights = def(0.f);
        auto ux_remapped = def(0.f);
        auto lower_sum = def(0.f);
        auto upper_sum = def(1.f);
        for (auto i = 0u; i < max_sampling_techique_count; i++) {
            auto sel = (_lobes & sampling_techniques[i]) != 0u & (u.x > sum_weights);
            sampling_tech = ite(sel, i, sampling_tech);
            lower_sum = ite(sel, sum_weights, lower_sum);
            sum_weights += _sampling_weights[i];
            upper_sum = ite(sel, sum_weights, upper_sum);
        }
        u.x = saturate((u.x - lower_sum) / (upper_sum - lower_sum));

        // sample
        auto wo_local = _it.wo_local();
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        std::array<const BxDF *, max_sampling_techique_count> lobes{
            _diffuse.get(), _specular.get(), _clearcoat.get(),
            _spec_trans.get(), _thin_spec_trans.get(), _diff_trans.get()};
        $switch(sampling_tech) {
            for (auto i = 0u; i < max_sampling_techique_count; i++) {
                $case(i) {
                    auto pdf = def<float>();
                    static_cast<void>(lobes[i]->sample(wo_local, &wi_local, u, &pdf));
                };
            }
            $default { unreachable(); };
        };
        auto eval = evaluate_local(wo_local, wi_local);
        auto wi = _it.shading().local_to_world(wi_local);
        return {.wi = std::move(wi), .eval = std::move(eval)};
    }

    void update() noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
    void backward(Pipeline &pipeline, Expr<float3> k, Float learning_rate, Expr<float3> wi) noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
};

luisa::unique_ptr<Surface::Closure> DisneySurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<Params>(it.shape()->surface_buffer_id()).read(0u);
    return luisa::make_unique<DisneySurfaceClosure>(
        it, swl,
        pipeline.evaluate_color_texture(params.color, it, swl, time),
        pipeline.evaluate_generic_texture(params.metallic, it, time).x,
        pipeline.evaluate_generic_texture(params.eta, it, time).x,
        pipeline.evaluate_generic_texture(params.roughness, it, time).x,
        pipeline.evaluate_generic_texture(params.specular_tint, it, time).x,
        pipeline.evaluate_generic_texture(params.anisotropic, it, time).x,
        pipeline.evaluate_generic_texture(params.sheen, it, time).x,
        pipeline.evaluate_generic_texture(params.sheen_tint, it, time).x,
        pipeline.evaluate_generic_texture(params.clearcoat, it, time).x,
        pipeline.evaluate_generic_texture(params.clearcoat_gloss, it, time).x,
        pipeline.evaluate_generic_texture(params.speculat_trans, it, time).x,
        pipeline.evaluate_generic_texture(params.flatness, it, time).x,
        pipeline.evaluate_generic_texture(params.diffuse_trans, it, time).x,
        params.thin);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DisneySurface)

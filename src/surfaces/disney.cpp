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

namespace detail {

using namespace compute;

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
//
// The Schlick Fresnel approximation is:
//
// R = R(0) + (1 - R(0)) (1 - cos theta)^5,
//
// where R(0) is the reflectance at normal indicence.
[[nodiscard]] inline Float SchlickWeight(Float cosTheta) noexcept {
    auto m = saturate(1.f - cosTheta);
    return (m * m) * (m * m) * m;
}

[[nodiscard]] inline auto FrSchlick(auto R0, Float cosTheta) noexcept {
    return lerp(R0, 1.f, SchlickWeight(cosTheta));
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
        wh = normalize(wi + wo);
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
};

// Specialized Fresnel function used for the specular component, based on
// a mixture between dielectric and the Schlick Fresnel approximation.
class DisneyFresnel final : public Fresnel {

private:
    Float4 R0;
    Float metallic;
    Float eta;

public:
    DisneyFresnel(Float4 R0, Float metallic, Float eta) noexcept
        : R0{std::move(R0)}, metallic{std::move(metallic)}, eta{std::move(eta)} {}
    [[nodiscard]] Float4 evaluate(Expr<float> cosI) const noexcept override {
        return lerp(fresnel_dielectric(cosI, 1.f, eta), FrSchlick(R0, cosI), metallic);
    }
};

struct DisneyMicrofacetDistribution final : public TrowbridgeReitzDistribution {
    explicit DisneyMicrofacetDistribution(Float2 alpha) noexcept
        : TrowbridgeReitzDistribution{std::move(alpha)} {}
    [[nodiscard]] Float G(Expr<float3> wo, Expr<float3> wi) const noexcept override {
        return G1(wo) * G1(wi);
    }
};

}// namespace detail

class DisneySurfaceClosure final : public Surface::Closure {

private:
    const Interaction &_it;
    const SampledWavelengths &_swl;

public:
    DisneySurfaceClosure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> color, Expr<float> metallic, Expr<float> eta, Expr<float> roughness,
        Expr<float> specular_tint, Expr<float> anisotropic, Expr<float> sheen, Expr<float> sheen_tint,
        Expr<float> clearcoat, Expr<float> clearcoat_gloss, Expr<float> specular_trans,
        Expr<float> flatness, Expr<float> diffuse_trans) noexcept
        : _it{it}, _swl{swl} {}
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

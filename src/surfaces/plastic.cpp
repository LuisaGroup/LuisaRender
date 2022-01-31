//
// Created by Mike Smith on 2022/1/9.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

class PlasticSurface final : public Surface {

public:
    struct Params {
        TextureHandle Kd;
        TextureHandle Ks;
        TextureHandle roughness;
        bool remap_roughness;
    };

private:
    const Texture *_kd;
    const Texture *_ks;
    const Texture *_roughness;
    bool _remap_roughness;

public:
    PlasticSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kd{scene->load_texture(desc->property_node_or_default(
              "Kd", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _ks{scene->load_texture(desc->property_node_or_default(
              "Ks", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _roughness{scene->load_texture(desc->property_node_or_default(
              "roughness", SceneNodeDesc::shared_default_texture("ConstGeneric")))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        if (_kd->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in PlasticSurface::Kd. [{}]",
                desc->source_location().string());
        }
        if (_ks->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in PlasticSurface::Ks. [{}]",
                desc->source_location().string());
        }
        if (_roughness->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in PlasticSurface::roughness. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint, const Shape *) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<Params>(1u);
        Params params{
            .Kd = *pipeline.encode_texture(command_buffer, _kd),
            .Ks = *pipeline.encode_texture(command_buffer, _ks),
            .roughness = *pipeline.encode_texture(command_buffer, _roughness),
            .remap_roughness = _remap_roughness};
        command_buffer << buffer_view.copy_from(&params)
                       << compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

}// namespace luisa::render

LUISA_STRUCT(
    luisa::render::PlasticSurface::Params,
    Kd, Ks, roughness, remap_roughness){};

namespace luisa::render {

class PlasticClosure final : public Surface::Closure {

private:
    const Interaction &_interaction;
    TrowbridgeReitzDistribution _distribution;
    FresnelDielectric _fresnel;
    LambertianReflection _lambert;
    MicrofacetReflection _microfacet;

public:
    PlasticClosure(const Interaction &it, Expr<float4> Kd, Expr<float4> Ks, Expr<float> alpha) noexcept
        : _interaction{it}, _distribution{make_float2(alpha)}, _fresnel{1.5f, 1.0f},
          _lambert{Kd}, _microfacet{Ks, &_distribution, &_fresnel} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto wi_local = _interaction.shading().world_to_local(wi);
        auto f_d = _lambert.evaluate(wo_local, wi_local);
        auto pdf_d = _lambert.pdf(wo_local, wi_local);
        auto f_s = _microfacet.evaluate(wo_local, wi_local);
        auto pdf_s = _microfacet.pdf(wo_local, wi_local);
        return {.f = f_d + f_s, .pdf = 0.5f * (pdf_d + pdf_s)};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto u = sampler.generate_2d();
        auto lobe = cast<int>(u.x >= .5f);
        u.x = ite(lobe == 0u, u.x * 2.0f, -2.0f * u.x + 2.0f);
        auto pdf = def(0.f);
        auto f = def<float4>();
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        $if(lobe == 0u) {// Lambert
            u.x *= 2.0f;
            f = _lambert.sample(wo_local, &wi_local, u, &pdf);
            f += _microfacet.evaluate(wo_local, wi_local);
            pdf = (pdf + _microfacet.pdf(wo_local, wi_local)) * .5f;
        }
        $else{// Microfacet
            u.x = fma(-2.0f, u.x, 2.0f);
            f = _microfacet.sample(wo_local, &wi_local, u, &pdf);
            f += _lambert.evaluate(wo_local, wi_local);
            pdf = (pdf + _lambert.pdf(wo_local, wi_local)) * .5f;
        };
        auto wi = _interaction.shading().local_to_world(wi_local);
        return {.wi = wi, .eval = {.f = f, .pdf = pdf}};
    }
};

luisa::unique_ptr<Surface::Closure> PlasticSurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<Params>(it.shape()->surface_buffer_id()).read(0u);
    auto Kd_max = def(0.0f);
    auto Ks_max = def(0.0f);
    auto Kd = pipeline.evaluate_color_texture(params.Kd, it, swl, time, &Kd_max);
    auto Ks = pipeline.evaluate_color_texture(params.Ks, it, swl, time, &Ks_max);
    auto roughness = saturate(pipeline.evaluate_generic_texture(params.roughness, it, time).x);
    auto alpha = ite(
        params.remap_roughness,
        TrowbridgeReitzDistribution::roughness_to_alpha(roughness),
        roughness);
    auto scale = 1.0f / max(Kd_max + Ks_max, 1.0f);
    return luisa::make_unique<PlasticClosure>(
        it, scale * Kd, scale * Ks, alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PlasticSurface)

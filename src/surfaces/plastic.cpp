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
        TextureHandle eta;
        bool remap_roughness;
        bool isotropic;
        bool dispersion;
    };

private:
    const Texture *_kd;
    const Texture *_ks;
    const Texture *_roughness;
    const Texture *_eta;
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
          _eta{scene->load_texture(desc->property_node_or_default(
              "eta", SceneNodeDesc::shared_default_texture("ConstGeneric")))},
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
        if (_eta->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in PlasticSurface::eta. [{}]",
                desc->source_location().string());
        }
        if (_eta->channels() == 2u) [[unlikely]] {
            LUISA_ERROR(
                "Invalid channel count {} "
                "for PlasticSurface::eta.",
                desc->source_location().string());
        }
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint, const Shape *) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<Params>(1u);
        Params params{
            .Kd = *pipeline.encode_texture(command_buffer, _kd),
            .Ks = *pipeline.encode_texture(command_buffer, _ks),
            .roughness = *pipeline.encode_texture(command_buffer, _roughness),
            .eta = *pipeline.encode_texture(command_buffer, _eta),
            .remap_roughness = _remap_roughness,
            .isotropic = _roughness->channels() == 1u,
            .dispersion = _eta->channels() != 1u};
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
    Kd, Ks, roughness, eta,
    remap_roughness, isotropic, dispersion){};

namespace luisa::render {

class PlasticClosure final : public Surface::Closure {

private:
    const Interaction &_interaction;
    const SampledWavelengths &_swl;
    TrowbridgeReitzDistribution _distribution;
    FresnelDielectric _fresnel;
    LambertianReflection _lambert;
    MicrofacetReflection _microfacet;
    Float _kd_ratio;

public:
    PlasticClosure(const Interaction &it, const SampledWavelengths &swl,
                   Expr<float4> eta, Expr<float4> Kd, Expr<float4> Ks,
                   Expr<float2> alpha, Expr<float> Kd_ratio) noexcept
        : _interaction{it}, _swl{swl}, _distribution{alpha}, _fresnel{eta, make_float4(1.0f)},
          _lambert{Kd}, _microfacet{Ks, &_distribution, &_fresnel}, _kd_ratio{Kd_ratio} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto wi_local = _interaction.shading().world_to_local(wi);
        auto f_d = _lambert.evaluate(wo_local, wi_local);
        auto pdf_d = _lambert.pdf(wo_local, wi_local);
        auto f_s = _microfacet.evaluate(wo_local, wi_local);
        auto pdf_s = _microfacet.pdf(wo_local, wi_local);
        return {.swl = _swl, .f = f_d + f_s,
                .pdf = lerp(pdf_s, pdf_d, _kd_ratio),
                .alpha = _distribution.alpha(),
                .eta = make_float4(1.f)};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto f = def<float4>();
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto lobe = cast<int>(u.x >= _kd_ratio);
        $if(lobe == 0u) {// Lambert
            u.x = u.x / _kd_ratio;
            f = _lambert.sample(wo_local, &wi_local, u, &pdf);
            f += _microfacet.evaluate(wo_local, wi_local);
            auto pdf_s = _microfacet.pdf(wo_local, wi_local);
            pdf = lerp(pdf_s, pdf, _kd_ratio);
        }
        $else {// Microfacet
            u.x = (u.x - _kd_ratio) / (1.f - _kd_ratio);
            f = _microfacet.sample(wo_local, &wi_local, u, &pdf);
            f += _lambert.evaluate(wo_local, wi_local);
            auto pdf_d = _lambert.pdf(wo_local, wi_local);
            pdf = lerp(pdf, pdf_d, _kd_ratio);
        };
        auto wi = _interaction.shading().local_to_world(wi_local);
        return {.wi = wi, .eval = {.swl = _swl, .f = f, .pdf = pdf,
                                   .alpha = _distribution.alpha(),
                                   .eta = make_float4(1.f)}};
    }

    void update() noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
    }
    void backward(Pipeline &pipeline, const SampledWavelengths &swl_fixed, Expr<float4> k, Float learning_rate, Expr<float3> wi) noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("unimplemented");
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
    auto r = pipeline.evaluate_generic_texture(params.roughness, it, time);
    auto e = pipeline.evaluate_generic_texture(params.eta, it, time);
    auto eta_basis = ite(params.dispersion, e.xyz(), ite(e.x == 0.f, 1.5f, e.x));
    auto roughness = ite(params.isotropic, r.xx(), r.xy());
    auto alpha = ite(
        params.remap_roughness,
        TrowbridgeReitzDistribution::roughness_to_alpha(roughness),
        roughness);
    auto scale = 1.0f / max(Kd_max + Ks_max, 1.0f);
    auto Kd_lum = swl.cie_y(Kd);
    auto Ks_lum = swl.cie_y(Ks);
    auto Kd_ratio = ite(Kd_lum == 0.f, 0.f, Kd_lum / (Kd_lum + Ks_lum));
    // interpolate eta using Cauchy's dispersion formula
    auto inv_bb = sqr(1.f / make_float3(700.0f, 546.1f, 435.8f));
    auto m = make_float3x3(make_float3(1.f), inv_bb, sqr(inv_bb));
    auto c = inverse(m) * eta_basis;
    auto inv_ll = sqr(1.f / swl.lambda());
    auto eta = make_float4(
        dot(c, make_float3(1.f, inv_ll.x, sqr(inv_ll.x))),
        dot(c, make_float3(1.f, inv_ll.y, sqr(inv_ll.y))),
        dot(c, make_float3(1.f, inv_ll.z, sqr(inv_ll.z))),
        dot(c, make_float3(1.f, inv_ll.w, sqr(inv_ll.w))));
    return luisa::make_unique<PlasticClosure>(
        it, swl, eta, scale * Kd, scale * Ks,
        alpha, clamp(Kd_ratio, .1f, .9f));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PlasticSurface)

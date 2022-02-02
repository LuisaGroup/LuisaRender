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

class GlassSurface final : public Surface {

public:
    struct Params {
        TextureHandle Kr;
        TextureHandle Kt;
        TextureHandle roughness;
        TextureHandle eta;
        bool remap_roughness;
        bool isotropic;
        bool dispersion;
    };

private:
    const Texture *_kr;
    const Texture *_kt;
    const Texture *_roughness;
    const Texture *_eta;
    bool _remap_roughness;

public:
    GlassSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kr{scene->load_texture(desc->property_node_or_default(
              "Kr", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _kt{scene->load_texture(desc->property_node_or_default(
              "Kt", SceneNodeDesc::shared_default_texture("ConstColor")))},
          _roughness{scene->load_texture(desc->property_node_or_default(
              "roughness", SceneNodeDesc::shared_default_texture("ConstGeneric")))},
          _eta{scene->load_texture(desc->property_node_or_default(
              "eta", SceneNodeDesc::shared_default_texture("ConstGeneric")))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", false)} {
        if (_kr->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in GlassSurface::Kr. [{}]",
                desc->source_location().string());
        }
        if (_kt->category() != Texture::Category::COLOR) [[unlikely]] {
            LUISA_ERROR(
                "Non-color textures are not "
                "allowed in GlassSurface::Kt. [{}]",
                desc->source_location().string());
        }
        if (_roughness->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in GlassSurface::roughness. [{}]",
                desc->source_location().string());
        }
        if (_eta->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in GlassSurface::eta. [{}]",
                desc->source_location().string());
        }
        if (_eta->channels() == 2u) [[unlikely]] {
            LUISA_ERROR(
                "Invalid channel count {} "
                "for GlassSurface::eta.",
                desc->source_location().string());
        }
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint, const Shape *) const noexcept override {
        auto [buffer_view, buffer_id] = pipeline.arena_buffer<Params>(1u);
        Params params{
            .Kr = *pipeline.encode_texture(command_buffer, _kr),
            .Kt = *pipeline.encode_texture(command_buffer, _kt),
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
    luisa::render::GlassSurface::Params,
    Kr, Kt, roughness, eta,
    remap_roughness, isotropic, dispersion){};

namespace luisa::render {

class GlassClosure final : public Surface::Closure {

private:
    const Interaction &_interaction;
    const SampledWavelengths &_swl;
    TrowbridgeReitzDistribution _distribution;
    FresnelDielectric _fresnel;
    MicrofacetReflection _refl;
    MicrofacetTransmission _trans;
    Float _kr_ratio;
    Bool _dispersion;

public:
    GlassClosure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> eta, Expr<float4> Kr, Expr<float4> Kt,
        Expr<float2> alpha, Expr<float> Kr_ratio) noexcept
        : _interaction{it}, _swl{swl}, _distribution{alpha}, _fresnel{make_float4(1.0f), eta},
          _refl{Kr, &_distribution, &_fresnel}, _trans{Kt, &_distribution, make_float4(1.f), eta},
          _kr_ratio{Kr_ratio}, _dispersion{!(eta.x == eta.y & eta.y == eta.z & eta.z == eta.w)} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto wi_local = _interaction.shading().world_to_local(wi);
        auto f = def<float4>();
        auto pdf = def(0.f);
        auto swl = _swl;
        auto t = saturate(abs(_fresnel.evaluate(cos_theta(wo_local)).x) * _kr_ratio);
        $if(same_hemisphere(wo_local, wi_local)) {
            f = _refl.evaluate(wo_local, wi_local) / t;
            pdf = _refl.pdf(wo_local, wi_local);
        }
        $else {
            f = _trans.evaluate(wo_local, wi_local) / (1.f - t);
            pdf = _trans.pdf(wo_local, wi_local);
            $if(_dispersion) { swl.terminate_secondary(); };
        };
        return {.swl = swl, .f = f, .pdf = pdf};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _interaction.wo_local();
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto f = def<float4>();
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto t = saturate(_fresnel.evaluate(cos_theta(wo_local)).x * _kr_ratio);
        auto lobe = cast<int>(u.x >= t);
        auto swl = _swl;
        $if(lobe == 0u) {// Reflection
            u.x = u.x / t;
            f = _refl.sample(wo_local, &wi_local, u, &pdf) / t;
        }
        $else {// Transmission
            u.x = (u.x - t) / (1.f - t);
            f = _trans.sample(wo_local, &wi_local, u, &pdf) / (1.f - t);
            $if(_dispersion) { swl.terminate_secondary(); };
        };
        auto wi = _interaction.shading().local_to_world(wi_local);
        return {.wi = wi, .eval = {.swl = swl, .f = f, .pdf = pdf}};
    }
};

luisa::unique_ptr<Surface::Closure> GlassSurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<Params>(it.shape()->surface_buffer_id()).read(0u);
    auto Kr = pipeline.evaluate_color_texture(params.Kr, it, swl, time);
    auto Kt = pipeline.evaluate_color_texture(params.Kt, it, swl, time);
    auto r = pipeline.evaluate_generic_texture(params.roughness, it, time);
    auto e = pipeline.evaluate_generic_texture(params.eta, it, time);
    auto eta_basis = ite(params.dispersion, e.xyz(), ite(e.x == 0.f, 1.5f, e.x));
    auto roughness = sqr(ite(params.isotropic, r.xx(), r.xy()));
    auto alpha = ite(
        params.remap_roughness,
        TrowbridgeReitzDistribution::roughness_to_alpha(roughness),
        roughness);
    auto Kr_lum = swl.cie_y(Kr);
    auto Kt_lum = swl.cie_y(Kt);
    auto Kr_ratio = ite(Kr_lum == 0.f, 0.f, Kr_lum / (Kr_lum + Kt_lum));
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
    return luisa::make_unique<GlassClosure>(
        it, swl, eta, Kr, Kt, alpha, Kr_ratio);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GlassSurface)

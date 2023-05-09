//
// Created by Mike Smith on 2022/11/23.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

// Plastic surface from the Tungsten renderer:
// https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/RoughPlasticBsdf.cpp
//
// Original license:
//
// Copyright (c) 2014 Benedikt Bitterli <benedikt.bitterli (at) gmail (dot) com>
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from
// the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute
// it freely, subject to the following restrictions:
//
//     1. The origin of this software must not be misrepresented; you
//        must not claim that you wrote the original software. If you
//        use this software in a product, an acknowledgment in the
//        product documentation would be appreciated but is not required.
//
//     2. Altered source versions must be plainly marked as such, and
//        must not be misrepresented as being the original software.
//
//     3. This notice may not be removed or altered from any source
//        distribution.

class PlasticSurface : public Surface {

private:
    const Texture *_kd;
    const Texture *_roughness;
    const Texture *_sigma_a;
    const Texture *_eta;
    const Texture *_thickness;
    bool _remap_roughness;

public:
    PlasticSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kd{scene->load_texture(desc->property_node_or_default("Kd"))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _sigma_a{scene->load_texture(desc->property_node_or_default("sigma_a"))},
          _eta{scene->load_texture(desc->property_node_or_default("eta"))},
          _thickness{scene->load_texture(desc->property_node_or_default("thickness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {}
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint properties() const noexcept override { return property_reflective; }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PlasticInstance : public Surface::Instance {

public:
    struct PlasticContext {
        Interaction it;
        SampledSpectrum kd;
        Float kd_weight;
        SampledSpectrum sigma_a;
        Float eta;
        Float2 roughness;
    };

private:
    const Texture::Instance *_kd;
    const Texture::Instance *_roughness;
    const Texture::Instance *_sigma_a;
    const Texture::Instance *_eta;
    const Texture::Instance *_thickness;

public:
    PlasticInstance(const Pipeline &pipeline, const Surface *surface,
                    const Texture::Instance *Kd, const Texture::Instance *roughness,
                    const Texture::Instance *sigma_a, const Texture::Instance *eta,
                    const Texture::Instance *thickness) noexcept
        : Surface::Instance{pipeline, surface},
          _kd{Kd}, _roughness{roughness}, _sigma_a{sigma_a},
          _eta{eta}, _thickness{thickness} {}

public:
    [[nodiscard]] uint make_closure(
        PolymorphicClosure<Surface::Function> &closure,
        luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
        Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> PlasticSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    auto sigma_a = pipeline.build_texture(command_buffer, _sigma_a);
    auto eta = pipeline.build_texture(command_buffer, _eta);
    auto thickness = pipeline.build_texture(command_buffer, _thickness);
    return luisa::make_unique<PlasticInstance>(
        pipeline, this, Kd, roughness, sigma_a, eta, thickness);
}

class PlasticFunction : public Surface::Function {

private:
    [[nodiscard]] static auto _substrate_weight(Expr<float> Fo, Expr<float> kd_weight) noexcept {
        auto w = kd_weight * (1.0f - Fo);
        return ite(w == 0.f, 0.f, w / (w + Fo));
    }

public:
    [[nodiscard]] static luisa::string identifier() noexcept { return LUISA_RENDER_PLUGIN_NAME; }

    [[nodiscard]] SampledSpectrum albedo(
        const std::any &ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = std::any_cast<PlasticInstance::PlasticContext>(ctx_wrapper);
        auto substrate = LambertianReflection(ctx.kd);
        return substrate.albedo();
    }
    [[nodiscard]] Float2 roughness(
        const std::any &ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = std::any_cast<PlasticInstance::PlasticContext>(ctx_wrapper);
        auto distribution = TrowbridgeReitzDistribution(ctx.roughness);
        return TrowbridgeReitzDistribution::alpha_to_roughness(distribution.alpha());
    }

    [[nodiscard]] Surface::Evaluation evaluate(
        const std::any &ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        auto ctx = std::any_cast<PlasticInstance::PlasticContext>(ctx_wrapper);
        auto &it = ctx.it;
        auto distribution = TrowbridgeReitzDistribution(ctx.roughness);
        auto fresnel = FresnelDielectric(1.0f, ctx.eta);
        auto coat = MicrofacetReflection(SampledSpectrum{swl.dimension(), 1.f}, &distribution, &fresnel);
        auto substrate = LambertianReflection(ctx.kd);

        auto wo_local = it.shading().world_to_local(wo);
        auto sign = ite(cos_theta(wo_local) < 0.f,
                        make_float3(1.f, 1.f, -1.f),
                        make_float3(1.f, 1.f, 1.f));
        wo_local *= sign;
        auto wi_local = sign * it.shading().world_to_local(wi);
        // specular
        auto f_coat = coat.evaluate(wo_local, wi_local, mode);
        auto pdf_coat = coat.pdf(wo_local, wi_local, mode);
        // diffuse
        auto eta = fresnel.eta_t();
        auto Fi = fresnel_dielectric(abs_cos_theta(wi_local), 1.f, eta);
        auto Fo = fresnel_dielectric(abs_cos_theta(wo_local), 1.f, eta);
        auto a = exp(-(1.f / abs_cos_theta(wi_local) + 1.f / abs_cos_theta(wo_local)) * ctx.sigma_a);
        auto f_diffuse = (1.f - Fi) * (1.f - Fo) * sqr(1.f / eta) * a *
                         substrate.evaluate(wo_local, wi_local, mode);
        auto pdf_diffuse = substrate.pdf(wo_local, wi_local, mode);
        auto substrate_weight = _substrate_weight(Fo, ctx.kd_weight);
        auto f = (f_coat + f_diffuse) * abs_cos_theta(wi_local);
        auto pdf = lerp(pdf_coat, pdf_diffuse, substrate_weight);
        return {.f = f, .pdf = pdf};
    }

    [[nodiscard]] Surface::Sample sample(
        const std::any &ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u, TransportMode mode) const noexcept override {
        auto ctx = std::any_cast<PlasticInstance::PlasticContext>(ctx_wrapper);
        auto &it = ctx.it;
        auto distribution = TrowbridgeReitzDistribution(ctx.roughness);
        auto fresnel = FresnelDielectric(1.0f, ctx.eta);
        auto coat = MicrofacetReflection(SampledSpectrum{swl.dimension(), 1.f}, &distribution, &fresnel);
        auto substrate = LambertianReflection(ctx.kd);

        auto wo_local = it.shading().world_to_local(wo);
        auto sign = ite(cos_theta(wo_local) < 0.f,
                        make_float3(1.f, 1.f, -1.f),
                        make_float3(1.f, 1.f, 1.f));
        wo_local *= sign;
        auto eta = fresnel.eta_t();
        auto Fo = fresnel_dielectric(abs_cos_theta(wo_local), 1.f, eta);
        auto substrate_weight = _substrate_weight(Fo, ctx.kd_weight);
        BxDF::SampledDirection wi_sample;
        $if(u_lobe < substrate_weight) {// samples diffuse
            wi_sample = substrate.sample_wi(wo_local, u, mode);
        }
        $else {// samples specular
            wi_sample = coat.sample_wi(wo_local, u, mode);
        };
        SampledSpectrum f{swl.dimension(), 0.f};
        auto pdf = def(0.f);
        auto wi = def(make_float3(0.f, 0.f, 1.f));
        $if(wi_sample.valid) {
            auto wi_local = wi_sample.wi;
            wi = it.shading().local_to_world(wi_sample.wi * sign);
            auto f_coat = coat.evaluate(wo_local, wi_local, mode);
            auto pdf_coat = coat.pdf(wo_local, wi_local, mode);
            // diffuse
            auto Fi = fresnel_dielectric(abs_cos_theta(wi_local), 1.f, eta);
            auto a = exp(-(1.f / abs_cos_theta(wi_local) + 1.f / abs_cos_theta(wo_local)) * ctx.sigma_a);
            auto ee = sqr(1.f / fresnel.eta_t());
            auto f_diffuse = (1.f - Fi) * (1.f - Fo) * sqr(1.f / eta) * a *
                             substrate.evaluate(wo_local, wi_local, mode);
            auto pdf_diffuse = substrate.pdf(wo_local, wi_local, mode);
            f = (f_coat + f_diffuse) * abs_cos_theta(wi_local);
            pdf = lerp(pdf_coat, pdf_diffuse, substrate_weight);
        };
        return {.eval = {.f = f, .pdf = pdf},
                .wi = wi,
                .event = Surface::event_reflect};
    }
};

uint PlasticInstance::make_closure(
    PolymorphicClosure<Surface::Function> &closure,
    luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
    Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {

    auto roughness = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(*it, swl, time);
        auto remap = node<PlasticSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        roughness = _roughness->node()->channels() == 1u ?
                        (remap ? make_float2(r2a(r.x)) : r.xx()) :
                        (remap ? r2a(r.xy()) : r.xy());
    }
    auto eta = (_eta ? _eta->evaluate(*it, swl, time).x : 1.5f) / eta_i;
    auto [Kd, Kd_lum] = _kd ? _kd->evaluate_albedo_spectrum(*it, swl, time) :
                              Spectrum::Decode::one(swl.dimension());
    auto [sigma_a, sigma_a_lum] = _sigma_a ? _sigma_a->evaluate_albedo_spectrum(*it, swl, time) :
                                             Spectrum::Decode::zero(swl.dimension());
    auto thickness = _thickness ? _thickness->evaluate(*it, swl, time).x : 1.f;
    auto scaled_sigma_a = sigma_a * thickness;
    auto average_transmittance = exp(-2.f * sigma_a_lum * thickness);
    // Difference from the Tungsten renderer:
    // We use the fitted polynomial to approximate the integrated
    // Fresnel reflectance, rather than compute it on the fly.
    auto diffuse_fresnel = fresnel_dielectric_integral(eta);

    auto ctx = PlasticContext{
        .it = *it,
        .kd = Kd / (1.f - Kd * diffuse_fresnel),
        .kd_weight = Kd_lum * average_transmittance,
        .sigma_a = sigma_a,
        .eta = eta,
        .roughness = roughness,
    };

    return closure.register_instance<PlasticFunction>(std::move(ctx));
}

//using NormalMapOpacityPlasticSurface = NormalMapWrapper<OpacitySurfaceWrapper<
//    PlasticSurface, PlasticInstance, PlasticClosure>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PlasticSurface)

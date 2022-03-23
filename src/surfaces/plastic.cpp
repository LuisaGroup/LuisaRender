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
              "Kd", SceneNodeDesc::shared_default_texture("Constant")))},
          _ks{scene->load_texture(desc->property_node_or_default(
              "Ks", SceneNodeDesc::shared_default_texture("Constant")))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _eta{scene->load_texture(desc->property_node_or_default("eta"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        if (_eta != nullptr) {
            if (_eta->channels() == 2u) [[unlikely]] {
                LUISA_ERROR(
                    "Invalid channel count {} "
                    "for PlasticSurface::eta.",
                    desc->source_location().string());
            }
        }
    }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PlasticInstance final : public Surface::Instance {

private:
    const Texture::Instance *_kd;
    const Texture::Instance *_ks;
    const Texture::Instance *_roughness;
    const Texture::Instance *_eta;

public:
    PlasticInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *Kd, const Texture::Instance *Ks,
        const Texture::Instance *roughness, const Texture::Instance *eta) noexcept
        : Surface::Instance{pipeline, surface},
          _kd{Kd}, _ks{Ks}, _roughness{roughness}, _eta{eta} {}
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> PlasticSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto Ks = pipeline.build_texture(command_buffer, _ks);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    auto eta = pipeline.build_texture(command_buffer, _eta);
    return luisa::make_unique<PlasticInstance>(
        pipeline, this, Kd, Ks, roughness, eta);
}

class PlasticClosure final : public Surface::Closure {

private:
    SampledSpectrum _eta_i;
    TrowbridgeReitzDistribution _distribution;
    FresnelDielectric _fresnel;
    LambertianReflection _lambert;
    MicrofacetReflection _microfacet;
    Float _kd_ratio;

public:
    PlasticClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time,
        const SampledSpectrum &eta, const SampledSpectrum &Kd, const SampledSpectrum &Ks,
        Expr<float2> alpha, Expr<float> Kd_ratio) noexcept
        : Surface::Closure{instance, it, swl, time}, _eta_i{swl.dimension(), 1.f},
          _distribution{alpha}, _fresnel{eta, _eta_i},
          _lambert{Kd}, _microfacet{Ks, &_distribution, &_fresnel}, _kd_ratio{Kd_ratio} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f_d = _lambert.evaluate(wo_local, wi_local);
        auto pdf_d = _lambert.pdf(wo_local, wi_local);
        auto f_s = _microfacet.evaluate(wo_local, wi_local);
        auto pdf_s = _microfacet.pdf(wo_local, wi_local);
        return {.f = f_d + f_s,
                .pdf = lerp(pdf_s, pdf_d, _kd_ratio),
                .alpha = _distribution.alpha(),
                .eta = _eta_i};
    }

    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _it.wo_local();
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        SampledSpectrum f{_swl.dimension(), 0.f};
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto lobe = cast<int>(u.x >= _kd_ratio);
        $if(lobe == 0u) {// Lambert
            u.x = u.x / _kd_ratio;
            f = _lambert.sample(wo_local, &wi_local, u, &pdf) +
                _microfacet.evaluate(wo_local, wi_local);
            auto pdf_s = _microfacet.pdf(wo_local, wi_local);
            pdf = lerp(pdf_s, pdf, _kd_ratio);
        }
        $else {// Microfacet
            u.x = (u.x - _kd_ratio) / (1.f - _kd_ratio);
            f = _microfacet.sample(wo_local, &wi_local, u, &pdf) +
                _lambert.evaluate(wo_local, wi_local);
            auto pdf_d = _lambert.pdf(wo_local, wi_local);
            pdf = lerp(pdf, pdf_d, _kd_ratio);
        };
        auto wi = _it.shading().local_to_world(wi_local);
        return {.wi = wi,
                .eval = {.f = f,
                         .pdf = pdf,
                         .alpha = _distribution.alpha(),
                         .eta = _eta_i}};
    }
    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        // TODO
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
};

luisa::unique_ptr<Surface::Closure> PlasticInstance::closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto Kd_rgb = _kd->evaluate(it, time).xyz();
    auto Ks_rgb = _ks->evaluate(it, time).xyz();
    auto Kd_max = max(max(Kd_rgb.x, Kd_rgb.y), Kd_rgb.z);
    auto Ks_max = max(max(Ks_rgb.x, Ks_rgb.y), Ks_rgb.z);
    auto scale = 1.f / max(Kd_max + Ks_max, 1.f);
    Kd_rgb *= scale;
    Ks_rgb *= scale;
    auto Kd_lum = srgb_to_cie_y(Kd_rgb);
    auto Ks_lum = srgb_to_cie_y(Ks_rgb);
    auto Kd = swl.albedo_from_srgb(Kd_rgb);
    auto Ks = swl.albedo_from_srgb(Ks_rgb);
    auto alpha = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, time);
        auto remap = node<PlasticSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    SampledSpectrum eta{swl.dimension(), 1.5f};
    if (_eta != nullptr) {
        if (_eta->node()->channels() == 1u) {
            auto e = _eta->evaluate(it, time).x;
            for (auto i = 0u; i < eta.dimension(); i++) { eta[i] = e; }
        } else {
            auto e = _eta->evaluate(it, time).xyz();
            auto inv_bb = sqr(1.f / make_float3(700.0f, 546.1f, 435.8f));
            auto m = make_float3x3(make_float3(1.f), inv_bb, sqr(inv_bb));
            auto c = inverse(m) * e;
            for (auto i = 0u; i < swl.dimension(); i++) {
                auto inv_ll = sqr(1.f / swl.lambda(i));
                eta[i] = c.x + c.y * inv_ll + c.z * sqr(inv_ll);
            }
        }
    }
    auto Kd_ratio = ite(Kd_lum == 0.f, 0.f, Kd_lum / (Kd_lum + Ks_lum));
    return luisa::make_unique<PlasticClosure>(
        this, it, swl, time, eta, Kd * scale, Ks * scale,
        alpha, clamp(Kd_ratio, .1f, .9f));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PlasticSurface)

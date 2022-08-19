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
          _kd{scene->load_texture(desc->property_node("Kd"))},
          _ks{scene->load_texture(desc->property_node("Ks"))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _eta{scene->load_texture(desc->property_node_or_default("eta"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        if (_eta != nullptr) {
            if (_eta->channels() == 2u || _eta->channels() == 4u) [[unlikely]] {
                LUISA_ERROR(
                    "Invalid channel count {} "
                    "for PlasticSurface::eta.",
                    desc->source_location().string());
            }
        }
        LUISA_RENDER_CHECK_ALBEDO_TEXTURE(PlasticSurface, kd);
        LUISA_RENDER_CHECK_ALBEDO_TEXTURE(PlasticSurface, ks);
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(PlasticSurface, roughness, 1);
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(PlasticSurface, eta, 1);
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
    [[nodiscard]] auto Kd() const noexcept { return _kd; }
    [[nodiscard]] auto Ks() const noexcept { return _ks; }
    [[nodiscard]] auto roughness() const noexcept { return _roughness; }
    [[nodiscard]] auto eta() const noexcept { return _eta; }

private:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> _closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> time) const noexcept override;
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
    Float _kd_ratio;
    luisa::unique_ptr<TrowbridgeReitzDistribution> _distribution;
    luisa::unique_ptr<FresnelDielectric> _fresnel;
    luisa::unique_ptr<LambertianReflection> _lambert;
    luisa::unique_ptr<MicrofacetReflection> _microfacet;

public:
    PlasticClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time,
        const SampledSpectrum &eta, const SampledSpectrum &Kd, const SampledSpectrum &Ks,
        Expr<float2> alpha, Expr<float> Kd_ratio) noexcept
        : Surface::Closure{instance, it, swl, time},
          _eta_i{swl.dimension(), 1.f}, _kd_ratio{Kd_ratio} {
        _distribution = luisa::make_unique<TrowbridgeReitzDistribution>(alpha);
        _fresnel = luisa::make_unique<FresnelDielectric>(_eta_i, eta);
        _lambert = luisa::make_unique<LambertianReflection>(Kd);
        _microfacet = luisa::make_unique<MicrofacetReflection>(
            Ks, _distribution.get(), _fresnel.get());
    }

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f_d = _lambert->evaluate(wo_local, wi_local);
        auto pdf_d = _lambert->pdf(wo_local, wi_local);
        auto f_s = _microfacet->evaluate(wo_local, wi_local);
        auto pdf_s = _microfacet->pdf(wo_local, wi_local);
        return {.f = (f_d + f_s) * abs_cos_theta(wi_local),
                .pdf = lerp(pdf_s, pdf_d, _kd_ratio),
                .roughness = _distribution->alpha(),
                .eta = _eta_i};
    }

    [[nodiscard]] Surface::Sample sample(Expr<float> u_lobe, Expr<float2> u) const noexcept override {
        auto wo_local = _it.wo_local();
        auto pdf = def(0.f);
        SampledSpectrum f{_swl.dimension()};
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        $if(u_lobe < _kd_ratio) {// Lambert
            auto f_d = _lambert->sample(wo_local, &wi_local, u, &pdf);
            auto f_s = _microfacet->evaluate(wo_local, wi_local);
            auto pdf_s = _microfacet->pdf(wo_local, wi_local);
            f = f_d + f_s;
            pdf = lerp(pdf_s, pdf, _kd_ratio);
        }
        $else {// Microfacet
            auto f_s = _microfacet->sample(wo_local, &wi_local, u, &pdf);
            auto f_d = _lambert->evaluate(wo_local, wi_local);
            auto pdf_d = _lambert->pdf(wo_local, wi_local);
            f = f_d + f_s;
            pdf = lerp(pdf, pdf_d, _kd_ratio);
        };
        auto wi = _it.shading().local_to_world(wi_local);
        return {.wi = wi,
                .eval = {.f = f * abs_cos_theta(wi_local),
                         .pdf = pdf,
                         .roughness = _distribution->alpha(),
                         .eta = _eta_i}};
    }
    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        auto _instance = instance<PlasticInstance>();
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);

        auto [Kd, Kd_lum] = _instance->Kd()->evaluate_albedo_spectrum(_it, _swl, _time);
        auto [Ks, Ks_lum] = _instance->Ks()->evaluate_albedo_spectrum(_it, _swl, _time);
        auto k0 = Kd_lum + Ks_lum;
        auto scale = 1.f / max(k0, 1.f);

        // Ks, Kd
        auto d_f_d = _lambert->backward(wo_local, wi_local, df);
        auto d_f_s = _microfacet->backward(wo_local, wi_local, df);
        _instance->Kd()->backward_albedo_spectrum(_it, _swl, _time, zero_if_any_nan(d_f_d.dR * (1.f / scale)));
        _instance->Ks()->backward_albedo_spectrum(_it, _swl, _time, zero_if_any_nan(d_f_s.dR * (1.f / scale)));

        // roughness
        if (auto roughness = _instance->roughness()) {
            auto remap = _instance->node<PlasticSurface>()->remap_roughness();
            auto r_f4 = roughness->evaluate(_it, _time);
            auto r = roughness->node()->channels() == 1u ? r_f4.xx() : r_f4.xy();

            auto grad_alpha_roughness = [](auto &&x) noexcept {
                return TrowbridgeReitzDistribution::grad_alpha_roughness(x);
            };
            auto d_r = d_f_s.dAlpha * (remap ? grad_alpha_roughness(r) : make_float2(1.f));
            auto d_r_f4 = roughness->node()->channels() == 1u ?
                              make_float4(d_r.x + d_r.y, 0.f, 0.f, 0.f) :
                              make_float4(d_r, 0.f, 0.f);
            auto roughness_grad_range = 5.f * (roughness->node()->range().y - roughness->node()->range().x);
            roughness->backward(_it, _time, ite(any(isnan(d_r_f4) || abs(d_r_f4) > roughness_grad_range), 0.f, d_r_f4));
        }
    }
};

luisa::unique_ptr<Surface::Closure> PlasticInstance::_closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    // alpha
    auto alpha = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, time);
        auto remap = node<PlasticSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }

    // eta
    SampledSpectrum eta{swl.dimension(), 1.5f};
    if (_eta != nullptr) {
        if (_eta->node()->channels() == 1u) {
            auto e = _eta->evaluate(it, time).x;
            for (auto i = 0u; i < eta.dimension(); i++) { eta[i] = e; }
        } else {
            auto e = _eta->evaluate(it, time).xyz();
            auto inv_bb = sqr(1.f / rgb_spectrum_peak_wavelengths);
            auto m = make_float3x3(make_float3(1.f), inv_bb, sqr(inv_bb));
            auto c = inverse(m) * e;
            for (auto i = 0u; i < swl.dimension(); i++) {
                auto inv_ll = sqr(1.f / swl.lambda(i));
                eta[i] = c.x + c.y * inv_ll + c.z * sqr(inv_ll);
            }
        }
    }

    // Kd, Ks
    auto [Kd, Kd_lum] = _kd->evaluate_albedo_spectrum(it, swl, time);
    auto [Ks, Ks_lum] = _ks->evaluate_albedo_spectrum(it, swl, time);
    auto scale = 1.f / max(Kd_lum + Ks_lum, 1.f);
    Kd *= scale;
    Ks *= scale;

    // Kd_ratio
    auto Kd_ratio = ite(Kd_lum <= 0.f, 0.f, Kd_lum / (Kd_lum + Ks_lum));
    return luisa::make_unique<PlasticClosure>(
        this, it, swl, time, eta, Kd, Ks,
        alpha, clamp(Kd_ratio, 0.1f, 0.9f));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PlasticSurface)

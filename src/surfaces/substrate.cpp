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

class SubstrateSurface final : public Surface {

private:
    const Texture *_kd;
    const Texture *_ks;
    const Texture *_roughness;
    bool _remap_roughness;

public:
    SubstrateSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kd{scene->load_texture(desc->property_node("Kd"))},
          _ks{scene->load_texture(desc->property_node("Ks"))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        LUISA_RENDER_CHECK_ALBEDO_TEXTURE(SubstrateSurface, kd);
        LUISA_RENDER_CHECK_ALBEDO_TEXTURE(SubstrateSurface, ks);
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(SubstrateSurface, roughness, 1);
    }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class SubstrateInstance final : public Surface::Instance {

private:
    const Texture::Instance *_kd;
    const Texture::Instance *_ks;
    const Texture::Instance *_roughness;

public:
    SubstrateInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *Kd, const Texture::Instance *Ks,
        const Texture::Instance *roughness) noexcept
        : Surface::Instance{pipeline, surface},
          _kd{Kd}, _ks{Ks}, _roughness{roughness} {}
    [[nodiscard]] auto Kd() const noexcept { return _kd; }
    [[nodiscard]] auto Ks() const noexcept { return _ks; }
    [[nodiscard]] auto roughness() const noexcept { return _roughness; }

private:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> SubstrateSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto Ks = pipeline.build_texture(command_buffer, _ks);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    return luisa::make_unique<SubstrateInstance>(
        pipeline, this, Kd, Ks, roughness);
}

class SubstrateClosure final : public Surface::Closure {

private:
    luisa::unique_ptr<TrowbridgeReitzDistribution> _distribution;
    luisa::unique_ptr<FresnelBlend> _blend;
    Float _eta_i;

public:
    SubstrateClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time, Expr<float> eta_i,
        const SampledSpectrum &Kd, const SampledSpectrum &Ks, Expr<float2> alpha, Expr<float> Kd_ratio) noexcept
        : Surface::Closure{instance, it, swl, time},
          _distribution{luisa::make_unique<TrowbridgeReitzDistribution>(alpha)},
          _blend{luisa::make_unique<FresnelBlend>(Kd, Ks, _distribution.get(), Kd_ratio)},
          _eta_i{eta_i} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _blend->evaluate(wo_local, wi_local);
        auto pdf = _blend->pdf(wo_local, wi_local);
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }

    [[nodiscard]] Surface::Sample sample(Expr<float> u_lobe, Expr<float2> u) const noexcept override {
        auto wo_local = _it.wo_local();
        auto pdf = def(0.f);
        auto wi_local = def(make_float3());
        // TODO: pass u_lobe to _blend->sample()
        auto f = _blend->sample(wo_local, &wi_local, u, &pdf);
        auto wi = _it.shading().local_to_world(wi_local);
        return {.eval = {.f = f * abs_cos_theta(wi_local), .pdf = pdf},
                .wi = wi,
                .eta = 1.f,
                .event = Surface::event_reflect};
    }

public:
    [[nodiscard]] Float2 roughness() const noexcept override { return _distribution->alpha(); }

private:
    void backward(Expr<float3> wi, const SampledSpectrum &df_in) const noexcept override {
        using compute::isinf;
        auto _instance = instance<SubstrateInstance>();
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto df = df_in * abs_cos_theta(wi_local);

        auto grad = _blend->backward(wo_local, wi_local, df);

        _instance->Kd()->backward_albedo_spectrum(_it, _swl, _time, zero_if_any_nan(grad.dRd));
        _instance->Ks()->backward_albedo_spectrum(_it, _swl, _time, zero_if_any_nan(grad.dRs));
        if (auto roughness = _instance->roughness()) {
            auto remap = _instance->node<SubstrateSurface>()->remap_roughness();
            auto r_f4 = roughness->evaluate(_it, _swl, _time);
            auto r = roughness->node()->channels() == 1u ? r_f4.xx() : r_f4.xy();

            auto grad_alpha_roughness = [](auto &&x) noexcept {
                return TrowbridgeReitzDistribution::grad_alpha_roughness(x);
            };
            auto d_r = grad.dAlpha * (remap ? grad_alpha_roughness(r) : make_float2(1.f));
            auto d_r_f4 = roughness->node()->channels() == 1u ?
                              make_float4(d_r.x + d_r.y, 0.f, 0.f, 0.f) :
                              make_float4(d_r, 0.f, 0.f);
            auto roughness_grad_range = 5.f * (roughness->node()->range().y - roughness->node()->range().x);
            roughness->backward(_it, _swl, _time,
                                ite(any(isnan(d_r_f4) || abs(d_r_f4) > roughness_grad_range), 0.f, d_r_f4));
        }
    }
};

luisa::unique_ptr<Surface::Closure> SubstrateInstance::closure(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> eta_i, Expr<float> time) const noexcept {
    auto [Kd, Kd_lum] = _kd->evaluate_albedo_spectrum(it, swl, time);
    auto [Ks, Ks_lum] = _ks->evaluate_albedo_spectrum(it, swl, time);
    auto alpha = def(make_float2(.5f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, swl, time);
        auto remap = node<SubstrateSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept {
            return TrowbridgeReitzDistribution::roughness_to_alpha(x);
        };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    //    auto cos_theta = dot(it.shading().n(), it.wo());
    //    auto pow5 = [](auto &&v) { return sqr(sqr(v)) * v; };
    auto Kd_ratio = Kd_lum / max(Kd_lum + Ks_lum, 1e-5f);
    return luisa::make_unique<SubstrateClosure>(
        this, it, swl, time, eta_i, Kd, Ks, alpha, Kd_ratio);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SubstrateSurface)

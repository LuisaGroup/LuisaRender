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
          _kd{scene->load_texture(desc->property_node_or_default(
              "Kd", SceneNodeDesc::shared_default_texture("Constant")))},
          _ks{scene->load_texture(desc->property_node_or_default(
              "Ks", SceneNodeDesc::shared_default_texture("Constant")))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {

        LUISA_RENDER_PARAM_CHANNEL_CHECK(SubstrateSurface, kd, >=, 3);
        LUISA_RENDER_PARAM_CHANNEL_CHECK(SubstrateSurface, ks, >=, 3);
        LUISA_RENDER_PARAM_CHANNEL_CHECK(SubstrateSurface, roughness, <=, 2);
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
    bool _remap_roughness;

public:
    SubstrateInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *Kd, const Texture::Instance *Ks,
        const Texture::Instance *roughness, bool remap_roughness) noexcept
        : Surface::Instance{pipeline, surface},
          _kd{Kd}, _ks{Ks}, _roughness{roughness}, _remap_roughness{remap_roughness} {}
    [[nodiscard]] auto Kd() const noexcept { return _kd; }
    [[nodiscard]] auto Ks() const noexcept { return _ks; }
    [[nodiscard]] auto roughness() const noexcept { return _roughness; }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }

private:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> _closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> SubstrateSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto Ks = pipeline.build_texture(command_buffer, _ks);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    return luisa::make_unique<SubstrateInstance>(
        pipeline, this, Kd, Ks, roughness, remap_roughness());
}

class SubstrateClosure final : public Surface::Closure {

private:
    luisa::unique_ptr<TrowbridgeReitzDistribution> _distribution;
    luisa::unique_ptr<FresnelBlend> _blend;
    SampledSpectrum _eta_i;

public:
    SubstrateClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time,
        const SampledSpectrum &Kd, const SampledSpectrum &Ks, Expr<float2> alpha) noexcept
        : Surface::Closure{instance, it, swl, time},
          _distribution{luisa::make_unique<TrowbridgeReitzDistribution>(alpha)},
          _blend{luisa::make_unique<FresnelBlend>(Kd, Ks, _distribution.get())},
          _eta_i{swl.dimension(), 1.f} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _blend->evaluate(wo_local, wi_local);
        auto pdf = _blend->pdf(wo_local, wi_local);
        return {.f = f,
                .pdf = pdf,
                .normal = _it.shading().n(),
                .roughness = _distribution->alpha(),
                .eta = _eta_i};
    }

    [[nodiscard]] Surface::Sample sample(Expr<float> u_lobe, Expr<float2> u) const noexcept override {
        auto wo_local = _it.wo_local();
        auto pdf = def(0.f);
        auto wi_local = def(make_float3());
        // TODO: pass u_lobe to _blend->sample()
        auto f = _blend->sample(wo_local, &wi_local, u, &pdf);
        auto wi = _it.shading().local_to_world(wi_local);
        return {.wi = wi,
                .eval = {.f = f,
                         .pdf = pdf,
                         .normal = _it.shading().n(),
                         .roughness = _distribution->alpha(),
                         .eta = _eta_i}};
    }

    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        auto _instance = instance<SubstrateInstance>();
        auto requires_grad_kd = _instance->Kd()->node()->requires_gradients(),
             requires_grad_ks = _instance->Ks()->node()->requires_gradients();
        $if(requires_grad_kd || requires_grad_ks) {
            auto wo_local = _it.wo_local();
            auto wi_local = _it.shading().world_to_local(wi);
            auto grad = _blend->backward(wo_local, wi_local, df);

            _instance->Kd()->backward_albedo_spectrum(_it, _swl, _time, grad.dRd);
            _instance->Ks()->backward_albedo_spectrum(_it, _swl, _time, grad.dRs);
            if (auto roughness = _instance->roughness()) {
                auto remap = _instance->remap_roughness();
                auto r_f4 = roughness->evaluate(_it, _time);
                auto r = roughness->node()->channels() == 1u ? r_f4.xx() : r_f4.xy();

                auto grad_alpha_roughness = [](auto &&x) noexcept {
                    return TrowbridgeReitzDistribution::grad_alpha_roughness(x);
                };
                auto d_r = grad.dAlpha * (remap ? grad_alpha_roughness(r) : make_float2(1.f));
                auto d_r_f4 = roughness->node()->channels() == 1u ?
                                  make_float4(d_r.x + d_r.y, 0.f, 0.f, 0.f) :
                                  make_float4(d_r.x, d_r.y, 0.f, 0.f);
                _instance->roughness()->backward(_it, _time, make_float4(1.f));
            }
        };
    }
};

luisa::unique_ptr<Surface::Closure> SubstrateInstance::_closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto Kd = _kd->evaluate_albedo_spectrum(it, swl, time);
    auto Ks = _ks->evaluate_albedo_spectrum(it, swl, time);
    auto alpha = def(make_float2(.5f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, time);
        auto remap = node<SubstrateSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept {
            return TrowbridgeReitzDistribution::roughness_to_alpha(x);
        };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    return luisa::make_unique<SubstrateClosure>(
        this, it, swl, time, Kd, Ks, alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SubstrateSurface)

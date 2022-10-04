//
// Created by Mike Smith on 2022/1/12.
//

#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class MirrorSurface : public Surface {

private:
    const Texture *_color;
    const Texture *_roughness;
    bool _remap_roughness;

public:
    MirrorSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _color{scene->load_texture(desc->property_node_or_default("color"))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        LUISA_RENDER_CHECK_ALBEDO_TEXTURE(MirrorSurface, color);
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(MirrorSurface, roughness, 1);
    }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint properties() const noexcept override { return property_reflective | property_differentiable; }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MirrorInstance : public Surface::Instance {

private:
    const Texture::Instance *_color;
    const Texture::Instance *_roughness;

public:
    MirrorInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *color, const Texture::Instance *roughness) noexcept
        : Surface::Instance{pipeline, surface},
          _color{color}, _roughness{roughness} {}
    [[nodiscard]] auto color() const noexcept { return _color; }
    [[nodiscard]] auto roughness() const noexcept { return _roughness; }

public:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MirrorSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto color = pipeline.build_texture(command_buffer, _color);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    return luisa::make_unique<MirrorInstance>(pipeline, this, color, roughness);
}

using namespace luisa::compute;

class SchlickFresnel : public Fresnel {

public:
    struct Gradient : public Fresnel::Gradient {
        SampledSpectrum dR0;
        explicit Gradient(const SampledSpectrum &dR0) noexcept : dR0{dR0} {}
    };

private:
    SampledSpectrum R0;

public:
    explicit SchlickFresnel(const SampledSpectrum &R0) noexcept : R0{R0} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosI) const noexcept override {
        auto m = saturate(1.f - cosI);
        auto weight = sqr(sqr(m)) * m;
        return R0.map([weight](auto, auto R) noexcept {
            return lerp(R, 1.f, weight);
        });
    }
    [[nodiscard]] luisa::unique_ptr<Fresnel::Gradient> backward(
        Expr<float> cosI, const SampledSpectrum &df) const noexcept override {
        auto m = saturate(1.f - cosI);
        auto weight = sqr(sqr(m)) * m;
        return luisa::make_unique<SchlickFresnel::Gradient>(df * weight);
    }
};

class MirrorClosure : public Surface::Closure {

private:
    luisa::unique_ptr<SchlickFresnel> _fresnel;
    luisa::unique_ptr<TrowbridgeReitzDistribution> _distribution;
    luisa::unique_ptr<MicrofacetReflection> _refl;

public:
    MirrorClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time,
        const SampledSpectrum &refl, Expr<float2> alpha) noexcept
        : Surface::Closure{instance, it, swl, time},
          _fresnel{luisa::make_unique<SchlickFresnel>(refl)},
          _distribution{luisa::make_unique<TrowbridgeReitzDistribution>(alpha)},
          _refl{luisa::make_unique<MicrofacetReflection>(refl, _distribution.get(), _fresnel.get())} {}

private:
    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        auto wo_local = _it.shading().world_to_local(wo);
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _refl->evaluate(wo_local, wi_local, mode);
        auto pdf = _refl->pdf(wo_local, wi_local, mode);
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }
    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo, Expr<float>, Expr<float2> u,
                                          TransportMode mode) const noexcept override {
        auto pdf = def(0.f);
        auto wo_local = _it.shading().world_to_local(wo);
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto f = _refl->sample(wo_local, &wi_local, u, &pdf, mode);
        return {.eval = {.f = f * abs_cos_theta(wi_local), .pdf = pdf},
                .wi = _it.shading().local_to_world(wi_local),
                .event = Surface::event_reflect};
    }
    void _backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df_in,
                   TransportMode mode) const noexcept override {
        auto _instance = instance<MirrorInstance>();
        auto wo_local = _it.shading().world_to_local(wo);
        auto wi_local = _it.shading().world_to_local(wi);
        auto df = df_in * abs_cos_theta(wi_local);
        auto grad = _refl->backward(wo_local, wi_local, df);
        auto d_fresnel = dynamic_cast<SchlickFresnel::Gradient *>(grad.dFresnel.get());
        if (auto color = _instance->color()) {
            color->backward_albedo_spectrum(_it, _swl, _time, zero_if_any_nan(grad.dR + d_fresnel->dR0));
        }
        if (auto roughness = _instance->roughness()) {
            auto remap = _instance->node<MirrorSurface>()->remap_roughness();
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

luisa::unique_ptr<Surface::Closure> MirrorInstance::closure(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> eta_i, Expr<float> time) const noexcept {
    auto alpha = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, swl, time);
        auto remap = node<MirrorSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept {
            return TrowbridgeReitzDistribution::roughness_to_alpha(x);
        };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    auto [color, _] = _color ? _color->evaluate_albedo_spectrum(it, swl, time) : Spectrum::Decode::one(swl.dimension());
    return luisa::make_unique<MirrorClosure>(this, it, swl, time, color, alpha);
}

using NormalMapOpacityMirrorSurface = NormalMapWrapper<OpacitySurfaceWrapper<
    MirrorSurface, MirrorInstance, MirrorClosure>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapOpacityMirrorSurface)

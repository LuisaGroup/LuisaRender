//
// Created by Mike Smith on 2022/1/12.
//

#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class MirrorSurface final : public Surface {

private:
    const Texture *_color;
    const Texture *_roughness;
    bool _remap_roughness;

public:
    MirrorSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _color{scene->load_texture(desc->property_node_or_default(
              "color", SceneNodeDesc::shared_default_texture("Constant")))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", false)} {}
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MirrorInstance final : public Surface::Instance {

private:
    const Texture::Instance *_color;
    const Texture::Instance *_roughness;

public:
    MirrorInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *color, const Texture::Instance *roughness) noexcept
        : Surface::Instance{pipeline, surface}, _color{color}, _roughness{roughness} {}

private:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> _closure(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MirrorSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto color = pipeline.build_texture(command_buffer, _color);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    return luisa::make_unique<MirrorInstance>(pipeline, this, color, roughness);
}

using namespace luisa::compute;

class SchlickFresnel final : public Fresnel {

public:
    struct Gradient {
        SampledSpectrum dR0;
    };

private:
    SampledSpectrum R0;

public:
    explicit SchlickFresnel(SampledSpectrum R0) noexcept : R0{std::move(R0)} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosI) const noexcept override {
        auto m = saturate(1.f - cosI);
        auto weight = sqr(sqr(m)) * m;
        return R0.map([weight](auto, auto R) noexcept {
            return lerp(R, 1.f, weight);
        });
    }
    [[nodiscard]] Gradient backward(Expr<float> cosI, const SampledSpectrum &dFr) const noexcept {
        // TODO
        LUISA_WARNING_WITH_LOCATION("Not implemented.");
    }
};

class MirrorClosure final : public Surface::Closure {

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
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _refl->evaluate(wo_local, wi_local);
        auto pdf = _refl->pdf(wo_local, wi_local);
        return {.f = f,
                .pdf = pdf,
                .normal = _it.shading().n(),
                .roughness = _distribution->alpha(),
                .eta = SampledSpectrum{_swl.dimension(), 1.f}};
    }
    [[nodiscard]] Surface::Sample sample(Expr<float>, Expr<float2> u) const noexcept override {
        auto pdf = def(0.f);
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto f = _refl->sample(_it.wo_local(), &wi_local, u, &pdf);
        return {.wi = _it.shading().local_to_world(wi_local),
                .eval = {.f = f,
                         .pdf = pdf,
                         .normal = _it.shading().n(),
                         .roughness = _distribution->alpha(),
                         .eta = SampledSpectrum{_swl.dimension(), 1.f}}};
    }

    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        // TODO
        LUISA_WARNING_WITH_LOCATION("Not implemented.");
    }
};

luisa::unique_ptr<Surface::Closure> MirrorInstance::_closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto alpha = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, time);
        auto remap = node<MirrorSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    auto color = _color->evaluate_albedo_spectrum(it, swl, time);
    return luisa::make_unique<MirrorClosure>(this, it, swl, time, color, alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MirrorSurface)

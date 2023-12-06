//
// Created by Mike Smith on 2022/1/9.
//

#include "core/logging.h"
#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

class MatteSurface : public Surface {

private:
    const Texture *_kd;
    const Texture *_sigma;

public:
    MatteSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kd{scene->load_texture(desc->property_node_or_default("Kd"))},
          _sigma{scene->load_texture(desc->property_node_or_default("sigma"))} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint properties() const noexcept override { return property_reflective | property_differentiable; }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MatteInstance : public Surface::Instance {

private:
    const Texture::Instance *_kd;
    const Texture::Instance *_sigma;

public:
    MatteInstance(const Pipeline &pipeline, const Surface *surface,
                  const Texture::Instance *Kd, const Texture::Instance *sigma) noexcept
        : Surface::Instance{pipeline, surface}, _kd{Kd}, _sigma{sigma} {}
    [[nodiscard]] auto Kd() const noexcept { return _kd; }
    [[nodiscard]] auto sigma() const noexcept { return _sigma; }

public:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(const SampledWavelengths &swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure *closure, const Interaction &it, Expr<float3> wo, Expr<float> eta_i) const noexcept override;
    [[nodiscard]] luisa::string closure_identifier() const noexcept override {
        return luisa::format("matte<{}, {}>",
                             Texture::Instance::diff_param_identifier(_kd),
                             Texture::Instance::diff_param_identifier(_sigma));
    }
};

luisa::unique_ptr<Surface::Instance> MatteSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto sigma = pipeline.build_texture(command_buffer, _sigma);
    return luisa::make_unique<MatteInstance>(pipeline, this, Kd, sigma);
}

class MatteClosure : public Surface::Closure {

public:
    struct Context {
        Interaction it;
        SampledSpectrum Kd;
        Float sigma;
    };

public:
    using Surface::Closure::Closure;

    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return context<Context>().Kd; }
    [[nodiscard]] Float2 roughness() const noexcept override { return make_float2(1.f); }
    [[nodiscard]] const Interaction &it() const noexcept override { return context<Context>().it; }

private:
    luisa::unique_ptr<OrenNayar> _refl;

public:
    void pre_eval() noexcept override {
        auto &ctx = context<Context>();
        _refl = luisa::make_unique<OrenNayar>(ctx.Kd, ctx.sigma);
    }
    void post_eval() noexcept override {
        _refl = nullptr;
    }

private:
    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo,
                                                Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        auto &&ctx = context<Context>();
        auto wo_local = ctx.it.shading().world_to_local(wo);
        auto wi_local = ctx.it.shading().world_to_local(wi);
        auto f = _refl->evaluate(wo_local, wi_local, mode);
        auto pdf = _refl->pdf(wo_local, wi_local, mode);
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }

    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo,
                                          Expr<float> u_lobe, Expr<float2> u,
                                          TransportMode mode) const noexcept override {
        auto &&ctx = context<Context>();
        auto wo_local = ctx.it.shading().world_to_local(wo);
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto pdf = def(0.f);
        auto f = _refl->sample(wo_local, std::addressof(wi_local),
                               u, std::addressof(pdf), mode);
        auto wi = ctx.it.shading().local_to_world(wi_local);
        return {.eval = {.f = f * abs_cos_theta(wi_local), .pdf = pdf},
                .wi = wi,
                .event = Surface::event_reflect};
    }

    void _backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df_in,
                   TransportMode mode) const noexcept override {
        auto &&ctx = context<Context>();
        auto _instance = instance<MatteInstance>();
        auto wo_local = ctx.it.shading().world_to_local(wo);
        auto wi_local = ctx.it.shading().world_to_local(wi);
        auto df = df_in * abs_cos_theta(wi_local);
        auto grad = _refl->backward(wo_local, wi_local, df);
        // device_log("grad in matte: ({}, {}, {})", grad.dR[0u], grad.dR[1u], grad.dR[2u]);
        if (auto kd = _instance->Kd()) {
            kd->backward_albedo_spectrum(ctx.it, swl(), time(), zero_if_any_nan(grad.dR));
        }
        if (auto sigma = _instance->sigma()) {
            auto dv = make_float4(ite(isnan(grad.dSigma), 0.f, grad.dSigma), 0.f, 0.f, 0.f);
            sigma->backward(ctx.it, swl(), time(), dv);
        }
    }
};

luisa::unique_ptr<Surface::Closure> MatteInstance::create_closure(
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return luisa::make_unique<MatteClosure>(this, pipeline(), swl, time);
}

void MatteInstance::populate_closure(Surface::Closure *closure, const Interaction &it,
                                     Expr<float3> wo, Expr<float> eta_i) const noexcept {
    auto &swl = closure->swl();
    auto time = closure->time();
    auto [Kd, _] = _kd ? _kd->evaluate_albedo_spectrum(it, swl, time) :
                         Spectrum::Decode::one(swl.dimension());
    auto sigma = _sigma && !_sigma->node()->is_black() ?
                     luisa::make_optional(saturate(_sigma->evaluate(it, swl, time).x) * 90.f) :
                     luisa::nullopt;

    MatteClosure::Context ctx{
        .it = it,
        .Kd = Kd,
        .sigma = sigma ? sigma.value() : 0.f};
    closure->bind(std::move(ctx));
}

using NormalMapOpacityMatteSurface = NormalMapWrapper<OpacitySurfaceWrapper<
    MatteSurface, MatteInstance>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapOpacityMatteSurface)

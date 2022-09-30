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

class MatteSurface : public Surface {

private:
    const Texture *_kd;
    const Texture *_sigma;

public:
    MatteSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _kd{scene->load_texture(desc->property_node_or_default("Kd"))},
          _sigma{scene->load_texture(desc->property_node_or_default("sigma"))} {
        LUISA_RENDER_CHECK_ALBEDO_TEXTURE(MatteSurface, kd);
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(MatteSurface, sigma, 1);
    }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
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
    MatteInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *Kd, const Texture::Instance *sigma) noexcept
        : Surface::Instance{pipeline, surface}, _kd{Kd}, _sigma{sigma} {}
    [[nodiscard]] auto Kd() const noexcept { return _kd; }
    [[nodiscard]] auto sigma() const noexcept { return _sigma; }

public:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MatteSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto sigma = pipeline.build_texture(command_buffer, _sigma);
    return luisa::make_unique<MatteInstance>(pipeline, this, Kd, sigma);
}

class MatteClosure : public Surface::Closure {

private:
    luisa::unique_ptr<OrenNayar> _oren_nayar;

public:
    MatteClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> time, SampledSpectrum albedo, Expr<float> sigma) noexcept
        : Surface::Closure{instance, it, swl, time},
          _oren_nayar{luisa::make_unique<OrenNayar>(std::move(albedo), sigma)} {}

private:
    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        auto wo_local = _it.shading().world_to_local(wo);
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _oren_nayar->evaluate(wo_local, wi_local, mode);
        auto pdf = _oren_nayar->pdf(wo_local, wi_local, mode);
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }
    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo, Expr<float>, Expr<float2> u,
                                          TransportMode mode) const noexcept override {
        auto wo_local = _it.shading().world_to_local(wo);
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto pdf = def(0.f);
        auto f = _oren_nayar->sample(wo_local, &wi_local, u, &pdf, mode);
        auto wi = _it.shading().local_to_world(wi_local);
        return {.eval = {.f = f * abs_cos_theta(wi_local), .pdf = pdf},
                .wi = wi,
                .event = Surface::event_reflect};
    }
    void _backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df_in,
                   TransportMode mode) const noexcept override {
        auto _instance = instance<MatteInstance>();
        auto wo_local = _it.shading().world_to_local(wo);
        auto wi_local = _it.shading().world_to_local(wi);
        auto df = df_in * abs_cos_theta(wi_local);
        auto grad = _oren_nayar->backward(wo_local, wi_local, df);
        if (auto kd = _instance->Kd()) {
            kd->backward_albedo_spectrum(_it, _swl, _time, zero_if_any_nan(grad.dR));
        }
        if (auto sigma = _instance->sigma()) {
            auto dv = make_float4(ite(isnan(grad.dSigma), 0.f, grad.dSigma), 0.f, 0.f, 0.f);
            sigma->backward(_it, _swl, _time, dv);
        }
    }
};

luisa::unique_ptr<Surface::Closure> MatteInstance::closure(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> eta_i, Expr<float> time) const noexcept {
    auto [Kd, _] = _kd ? _kd->evaluate_albedo_spectrum(it, swl, time) : Spectrum::Decode::one(swl.dimension());
    auto sigma = _sigma ? clamp(_sigma->evaluate(it, swl, time).x, 0.f, 90.f) : 0.f;
    return luisa::make_unique<MatteClosure>(this, it, swl, time, Kd, sigma);
}

using NormalMapOpacityMatteSurface = NormalMapMixin<OpacitySurfaceMixin<
    MatteSurface, MatteInstance, MatteClosure>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapOpacityMatteSurface)

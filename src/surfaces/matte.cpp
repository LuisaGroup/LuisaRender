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
          _sigma{scene->load_texture(desc->property_node_or_default("sigma"))} {}
    [[nodiscard]] luisa::string closure_identifier() const noexcept override { return luisa::string(impl_type()); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint properties() const noexcept override { return property_reflective; }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MatteInstance : public Surface::Instance {

public:
    struct MatteContext {
        Interaction it;
        SampledSpectrum kd;
        Float sigma;
    };

private:
    const Texture::Instance *_kd;
    const Texture::Instance *_sigma;

public:
    MatteInstance(
        const Pipeline &pipeline, const Surface *surface,
        const Texture::Instance *Kd, const Texture::Instance *sigma) noexcept
        : Surface::Instance{pipeline, surface}, _kd{Kd}, _sigma{sigma} {}

public:
    [[nodiscard]] uint make_closure(
        PolymorphicClosure<Surface::Function> &closure,
        luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
        Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MatteSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto sigma = pipeline.build_texture(command_buffer, _sigma);
    return luisa::make_unique<MatteInstance>(pipeline, this, Kd, sigma);
}

class MatteFunction : public Surface::Function {

public:
    [[nodiscard]] SampledSpectrum albedo(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<MatteInstance::MatteContext>();
        auto refl = OrenNayar{ctx->kd, ctx->sigma};
        return refl.albedo();
    }
    [[nodiscard]] Float2 roughness(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        return make_float2(1.f);
    }

    [[nodiscard]] Surface::Evaluation evaluate(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        auto ctx = ctx_wrapper->data<MatteInstance::MatteContext>();
        auto &it = ctx->it;
        auto refl = OrenNayar{ctx->kd, ctx->sigma};

        auto wo_local = it.shading().world_to_local(wo);
        auto wi_local = it.shading().world_to_local(wi);
        auto f = refl.evaluate(wo_local, wi_local, mode);
        auto pdf = refl.pdf(wo_local, wi_local, mode);
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }
    [[nodiscard]] Surface::Sample sample(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float>, Expr<float2> u, TransportMode mode) const noexcept override {
        auto ctx = ctx_wrapper->data<MatteInstance::MatteContext>();
        auto &it = ctx->it;
        auto refl = OrenNayar{ctx->kd, ctx->sigma};

        auto wo_local = it.shading().world_to_local(wo);
        auto wi_local = def(make_float3(0.0f, 0.0f, 1.0f));
        auto pdf = def(0.f);
        auto f = refl.sample(wo_local, std::addressof(wi_local),
                             u, std::addressof(pdf), mode);
        auto wi = it.shading().local_to_world(wi_local);
        return {.eval = {.f = f * abs_cos_theta(wi_local), .pdf = pdf},
                .wi = wi,
                .event = Surface::event_reflect};
    }
};

uint MatteInstance::make_closure(
    PolymorphicClosure<Surface::Function> &closure,
    luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
    Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {
    auto [Kd, _] = _kd ? _kd->evaluate_albedo_spectrum(*it, swl, time) :
                         Spectrum::Decode::one(swl.dimension());
    auto sigma = _sigma && !_sigma->node()->is_black() ?
                     luisa::make_optional(saturate(_sigma->evaluate(*it, swl, time).x) * 90.f) :
                     luisa::nullopt;

    auto ctx = MatteContext{
        .it = *it,
        .kd = Kd,
        .sigma = sigma ? sigma.value() : 0.f};

    auto [tag, slot] = closure.register_instance<MatteFunction>(node()->closure_identifier());
    slot->create_data(std::move(ctx));
    return tag;
}

//using NormalMapOpacityMatteSurface = NormalMapWrapper<OpacitySurfaceWrapper<
//    MatteSurface, MatteInstance, MatteFunction>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MatteSurface)

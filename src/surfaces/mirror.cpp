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
          _color{scene->load_texture(desc->property_node_or_default(
              "color", lazy_construct([desc] {
                  return desc->property_node_or_default("Kd");
              })))},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {}
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::string closure_identifier() const noexcept override { return luisa::string(impl_type()); }
    [[nodiscard]] uint properties() const noexcept override { return property_reflective; }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MirrorInstance : public Surface::Instance {

public:
    struct MirrorContext {
        Interaction it;
        SampledSpectrum refl;
        Float2 alpha;
    };

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
    [[nodiscard]] uint make_closure(
        PolymorphicClosure<Surface::Function> &closure,
        luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
        Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MirrorSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto color = pipeline.build_texture(command_buffer, _color);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    return luisa::make_unique<MirrorInstance>(pipeline, this, color, roughness);
}

using namespace luisa::compute;

class SchlickFresnel : public Fresnel {

private:
    SampledSpectrum R0;

public:
    explicit SchlickFresnel(const SampledSpectrum &R0) noexcept : R0{R0} {}
    [[nodiscard]] SampledSpectrum evaluate(Expr<float> cosI) const noexcept override {
        auto m = saturate(1.f - cosI);
        auto weight = sqr(sqr(m)) * m;
        return (1.f - weight) * R0 + weight;
    }
};

class MirrorFunction : public Surface::Function {

public:
    [[nodiscard]] static luisa::string identifier() noexcept { return LUISA_RENDER_PLUGIN_NAME; }

    [[nodiscard]] SampledSpectrum albedo(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<MirrorInstance::MirrorContext>();
        auto fresnel = SchlickFresnel(ctx->refl);
        auto distribution = TrowbridgeReitzDistribution(ctx->alpha);
        auto refl = MicrofacetReflection(ctx->refl, &distribution, &fresnel);
        return refl.albedo();
    }
    [[nodiscard]] Float2 roughness(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<MirrorInstance::MirrorContext>();
        auto distribution = TrowbridgeReitzDistribution(ctx->alpha);
        return TrowbridgeReitzDistribution::alpha_to_roughness(distribution.alpha());
    }
    [[nodiscard]] const Interaction *it(
        const Surface::FunctionContext *ctx_wrapper, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<MirrorInstance::MirrorContext>();
        return &ctx->it;
    }

    [[nodiscard]] Surface::Evaluation evaluate(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        auto ctx = ctx_wrapper->data<MirrorInstance::MirrorContext>();
        auto &it = ctx->it;
        auto fresnel = SchlickFresnel(ctx->refl);
        auto distribution = TrowbridgeReitzDistribution(ctx->alpha);
        auto refl = MicrofacetReflection(ctx->refl, &distribution, &fresnel);

        auto wo_local = it.shading().world_to_local(wo);
        auto wi_local = it.shading().world_to_local(wi);
        auto f = refl.evaluate(wo_local, wi_local, mode);
        auto pdf = refl.pdf(wo_local, wi_local, mode);
        return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    }
    [[nodiscard]] Surface::Sample sample(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float>, Expr<float2> u, TransportMode mode) const noexcept override {
        auto ctx = ctx_wrapper->data<MirrorInstance::MirrorContext>();
        auto &it = ctx->it;
        auto fresnel = SchlickFresnel(ctx->refl);
        auto distribution = TrowbridgeReitzDistribution(ctx->alpha);
        auto refl = MicrofacetReflection(ctx->refl, &distribution, &fresnel);

        auto pdf = def(0.f);
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto wo_local = it.shading().world_to_local(wo);
        auto f = refl.sample(wo_local, std::addressof(wi_local),
                               u, std::addressof(pdf), mode);
        auto wi = it.shading().local_to_world(wi_local);
        return {.eval = {.f = f * abs_cos_theta(wi_local), .pdf = pdf},
                .wi = wi,
                .event = Surface::event_reflect};
    }
};

uint MirrorInstance::make_closure(
    PolymorphicClosure<Surface::Function> &closure,
    luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
    Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {
    auto alpha = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(*it, swl, time);
        auto remap = node<MirrorSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept {
            return TrowbridgeReitzDistribution::roughness_to_alpha(x);
        };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    auto [color, _] = _color ? _color->evaluate_albedo_spectrum(*it, swl, time) :
                               Spectrum::Decode::one(swl.dimension());
    auto ctx = MirrorContext{
        .it = *it,
        .refl = color,
        .alpha = alpha
    };

    auto [tag, slot] = closure.register_instance<MirrorFunction>(node()->closure_identifier());
    slot->create_data(std::move(ctx));
    return tag;
}

//using NormalMapOpacityMirrorSurface = NormalMapWrapper<OpacitySurfaceWrapper<
//    MirrorSurface, MirrorInstance, MirrorClosure>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MirrorSurface)

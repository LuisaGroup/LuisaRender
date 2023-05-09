//
// Created by Mike Smith on 2022/3/27.
//

#include <base/surface.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class MixSurface : public Surface {

private:
    const Surface *_a;
    const Surface *_b;
    const Texture *_ratio;

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

public:
    MixSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _a{scene->load_surface(desc->property_node("a"))},
          _b{scene->load_surface(desc->property_node("b"))},
          _ratio{scene->load_texture(desc->property_node_or_default("ratio"))} {
        LUISA_ASSERT(!_a->is_null() && !_b->is_null(), "MixSurface: Both surfaces must be valid.");
        auto prop = _a->properties() | _b->properties();
        LUISA_ASSERT(!((prop & property_thin) && (prop & property_transmissive)),
                     "MixSurface: Cannot mix thin and transmissive surfaces.");
    }
    [[nodiscard]] luisa::string closure_identifier() const noexcept override {
        return luisa::format("{}<{}, {}>", impl_type(), _a->closure_identifier(), _b->closure_identifier());
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint properties() const noexcept override { return _a->properties() | _b->properties(); }
};

class MixSurfaceInstance : public Surface::Instance {

public:
    struct MixSurfaceContext {
        Float ratio;
    };

private:
    luisa::unique_ptr<Surface::Instance> _a;
    luisa::unique_ptr<Surface::Instance> _b;
    const Texture::Instance *_ratio;

public:
    MixSurfaceInstance(
        const Pipeline &pipeline, const MixSurface *surface, const Texture::Instance *ratio,
        luisa::unique_ptr<Surface::Instance> a, luisa::unique_ptr<Surface::Instance> b) noexcept
        : Surface::Instance{pipeline, surface},
          _a{std::move(a)}, _b{std::move(b)}, _ratio{ratio} {}
    [[nodiscard]] auto ratio() const noexcept { return _ratio; }

public:
    [[nodiscard]] uint make_closure(
        PolymorphicClosure<Surface::Function> &closure,
        luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
        Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MixSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto a = _a->build(pipeline, command_buffer);
    auto b = _b->build(pipeline, command_buffer);
    auto ratio = pipeline.build_texture(command_buffer, _ratio);
    return luisa::make_unique<MixSurfaceInstance>(
        pipeline, this, ratio, std::move(a), std::move(b));
}

class MixSurfaceFunction : public Surface::Function {

private:
    [[nodiscard]] static auto _mix(const Surface::Evaluation &eval_a,
                                   const Surface::Evaluation &eval_b, Expr<float> ratio) noexcept {
        auto t = 1.f - ratio;
        return Surface::Evaluation{
            .f = lerp(eval_a.f, eval_b.f, t),
            .pdf = lerp(eval_a.pdf, eval_b.pdf, t)};
    }

public:
    [[nodiscard]] SampledSpectrum albedo(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<MixSurfaceInstance::MixSurfaceContext>();

        auto function_a = ctx_wrapper->nested("a").function(0u);
        auto ctx_wrapper_a = ctx_wrapper->nested("a").context(0u);
        auto function_b = ctx_wrapper->nested("b").function(0u);
        auto ctx_wrapper_b = ctx_wrapper->nested("b").context(0u);

        auto albedo_a = function_a->albedo(ctx_wrapper_a, swl, time);
        auto albedo_b = function_b->albedo(ctx_wrapper_b, swl, time);
        return albedo_a * ctx->ratio + albedo_b * (1.f - ctx->ratio);
    }
    [[nodiscard]] Float2 roughness(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<MixSurfaceInstance::MixSurfaceContext>();

        auto function_a = ctx_wrapper->nested("a").function(0u);
        auto ctx_wrapper_a = ctx_wrapper->nested("a").context(0u);
        auto function_b = ctx_wrapper->nested("b").function(0u);
        auto ctx_wrapper_b = ctx_wrapper->nested("b").context(0u);

        auto roughness_a = function_a->roughness(ctx_wrapper_a, swl, time);
        auto roughness_b = function_b->roughness(ctx_wrapper_b, swl, time);

        return lerp(roughness_b, roughness_a, ctx->ratio);
    }

    [[nodiscard]] luisa::optional<Float> opacity(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<MixSurfaceInstance::MixSurfaceContext>();

        auto function_a = ctx_wrapper->nested("a").function(0u);
        auto ctx_wrapper_a = ctx_wrapper->nested("a").context(0u);
        auto function_b = ctx_wrapper->nested("b").function(0u);
        auto ctx_wrapper_b = ctx_wrapper->nested("b").context(0u);

        auto opacity_a = function_a->opacity(ctx_wrapper_a, swl, time);
        auto opacity_b = function_b->opacity(ctx_wrapper_b, swl, time);
        if (!opacity_a && !opacity_b) { return luisa::nullopt; }
        return lerp(opacity_b.value_or(1.f), opacity_a.value_or(1.f), ctx->ratio);
    }
    [[nodiscard]] luisa::optional<Float> eta(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<MixSurfaceInstance::MixSurfaceContext>();

        auto function_a = ctx_wrapper->nested("a").function(0u);
        auto ctx_wrapper_a = ctx_wrapper->nested("a").context(0u);
        auto function_b = ctx_wrapper->nested("b").function(0u);
        auto ctx_wrapper_b = ctx_wrapper->nested("b").context(0u);

        auto eta_a = function_a->eta(ctx_wrapper_a, swl, time);
        auto eta_b = function_b->eta(ctx_wrapper_b, swl, time);
        if (!eta_a) { return eta_b; }
        if (!eta_b) { return eta_a; }
        return lerp(*eta_b, *eta_a, ctx->ratio);
    }
    [[nodiscard]] luisa::optional<Bool> is_dispersive(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto function_a = ctx_wrapper->nested("a").function(0u);
        auto ctx_wrapper_a = ctx_wrapper->nested("a").context(0u);
        auto function_b = ctx_wrapper->nested("b").function(0u);
        auto ctx_wrapper_b = ctx_wrapper->nested("b").context(0u);

        auto a_dispersive = function_a->is_dispersive(ctx_wrapper_a, swl, time);
        auto b_dispersive = function_b->is_dispersive(ctx_wrapper_b, swl, time);
        if (!a_dispersive) { return b_dispersive; }
        if (!b_dispersive) { return a_dispersive; }
        return *a_dispersive | *b_dispersive;
    }

private:
    [[nodiscard]] Surface::Evaluation evaluate(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        auto ctx = ctx_wrapper->data<MixSurfaceInstance::MixSurfaceContext>();

        auto function_a = ctx_wrapper->nested("a").function(0u);
        auto ctx_wrapper_a = ctx_wrapper->nested("a").context(0u);
        auto eval_a = function_a->evaluate(ctx_wrapper_a, swl, time, wo, wi, mode);

        auto function_b = ctx_wrapper->nested("b").function(0u);
        auto ctx_wrapper_b = ctx_wrapper->nested("b").context(0u);
        auto eval_b = function_b->evaluate(ctx_wrapper_b, swl, time, wo, wi, mode);

        return _mix(eval_a, eval_b, ctx->ratio);
    }
    [[nodiscard]] Surface::Sample sample(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u, TransportMode mode) const noexcept override {
        auto ctx = ctx_wrapper->data<MixSurfaceInstance::MixSurfaceContext>();

        auto function_a = ctx_wrapper->nested("a").function(0u);
        auto ctx_wrapper_a = ctx_wrapper->nested("a").context(0u);
        auto function_b = ctx_wrapper->nested("b").function(0u);
        auto ctx_wrapper_b = ctx_wrapper->nested("b").context(0u);

        auto sample = Surface::Sample::zero(swl.dimension());
        $if(u_lobe < ctx->ratio) {// sample a
            auto sample_a = function_a->sample(ctx_wrapper_a, swl, time, wo, u_lobe / ctx->ratio, u, mode);
            auto eval_b = function_b->evaluate(ctx_wrapper_b, swl, time, wo, sample_a.wi, mode);
            sample.eval = _mix(sample_a.eval, eval_b, ctx->ratio);
            sample.wi = sample_a.wi;
            sample.event = sample_a.event;
        }
        $else {// sample b
            auto sample_b = function_b->sample(ctx_wrapper_b, swl, time, wo, (u_lobe - ctx->ratio) / (1.f - ctx->ratio), u, mode);
            auto eval_a = function_a->evaluate(ctx_wrapper_a, swl, time, wo, sample_b.wi, mode);
            sample.eval = _mix(eval_a, sample_b.eval, ctx->ratio);
            sample.wi = sample_b.wi;
            sample.event = sample_b.event;
        };
        return sample;
    }
};

uint MixSurfaceInstance::make_closure(
    PolymorphicClosure<Surface::Function> &closure,
    luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
    Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {
    auto ratio = _ratio == nullptr ? 0.5f : clamp(_ratio->evaluate(*it, swl, time).x, 0.f, 1.f);
    auto ctx = MixSurfaceContext{
        .ratio = ratio};

    auto [tag, slot] = closure.register_instance<MixSurfaceFunction>(node()->closure_identifier());
    slot->create_data(std::move(ctx));
    slot->create_nested("a");
    slot->create_nested("b");
    _a->make_closure(slot->nested("a"), it, swl, wo, eta_i, time);
    _b->make_closure(slot->nested("b"), it, swl, wo, eta_i, time);
    return tag;
}

//using NormalMapMixSurface = NormalMapWrapper<
//    MixSurface, MixSurfaceInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MixSurface)

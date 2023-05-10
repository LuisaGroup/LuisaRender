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
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::string closure_identifier() const noexcept override {
        return luisa::format("{}<{}, {}>", impl_type(), _a->closure_identifier(), _b->closure_identifier());
    }
    [[nodiscard]] uint properties() const noexcept override { return _a->properties() | _b->properties(); }
};

class MixSurfaceInstance : public Surface::Instance {

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
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(const SampledWavelengths &swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure *closure, const Interaction &it, Expr<float3> wo, Expr<float> eta_i) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MixSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto a = _a->build(pipeline, command_buffer);
    auto b = _b->build(pipeline, command_buffer);
    auto ratio = pipeline.build_texture(command_buffer, _ratio);
    return luisa::make_unique<MixSurfaceInstance>(
        pipeline, this, ratio, std::move(a), std::move(b));
}

class MixSurfaceClosure : public Surface::Closure {

private:
    luisa::unique_ptr<Surface::Closure> _a;
    luisa::unique_ptr<Surface::Closure> _b;

public:
    struct Context {
        Interaction it;
        Float ratio;
    };

private:
    [[nodiscard]] static auto _mix(const Surface::Evaluation &eval_a,
                                   const Surface::Evaluation &eval_b, Expr<float> ratio) noexcept {
        auto t = 1.f - ratio;
        return Surface::Evaluation{
            .f = lerp(eval_a.f, eval_b.f, t),
            .pdf = lerp(eval_a.pdf, eval_b.pdf, t)};
    }

public:
    using Surface::Closure::Closure;

    explicit MixSurfaceClosure(
        const Pipeline &pipeline,
        const SampledWavelengths &swl,
        Expr<float> time,
        luisa::unique_ptr<Surface::Closure> a, luisa::unique_ptr<Surface::Closure> b) noexcept
        : Surface::Closure(pipeline, swl, time), _a{std::move(a)}, _b{std::move(b)} {}
    [[nodiscard]] auto a() const noexcept { return _a.get(); }
    [[nodiscard]] auto b() const noexcept { return _b.get(); }
    void before_evaluation() noexcept override {
        _a->before_evaluation();
        _b->before_evaluation();
    }
    void after_evaluation() noexcept override {
        _a->after_evaluation();
        _b->after_evaluation();
    }

public:
    [[nodiscard]] SampledSpectrum albedo() const noexcept override {
        auto &&ctx = context<Context>();

        auto albedo_a = a()->albedo();
        auto albedo_b = b()->albedo();
        return albedo_a * ctx.ratio + albedo_b * (1.f - ctx.ratio);
    }
    [[nodiscard]] Float2 roughness() const noexcept override {
        auto &&ctx = context<Context>();

        auto roughness_a = a()->roughness();
        auto roughness_b = b()->roughness();

        return lerp(roughness_b, roughness_a, ctx.ratio);
    }

    [[nodiscard]] const Interaction &it() const noexcept override { return context<Context>().it; }
    [[nodiscard]] luisa::optional<Float> opacity() const noexcept override {
        auto &&ctx = context<Context>();

        auto opacity_a = a()->opacity();
        auto opacity_b = b()->opacity();
        if (!opacity_a && !opacity_b) { return luisa::nullopt; }
        return lerp(opacity_b.value_or(1.f), opacity_a.value_or(1.f), ctx.ratio);
    }
    [[nodiscard]] luisa::optional<Float> eta() const noexcept override {
        auto &&ctx = context<Context>();

        auto eta_a = a()->eta();
        auto eta_b = b()->eta();
        if (!eta_a) { return eta_b; }
        if (!eta_b) { return eta_a; }
        return lerp(*eta_b, *eta_a, ctx.ratio);
    }
    [[nodiscard]] luisa::optional<Bool> is_dispersive() const noexcept override {
        auto a_dispersive = a()->is_dispersive();
        auto b_dispersive = b()->is_dispersive();
        if (!a_dispersive) { return b_dispersive; }
        if (!b_dispersive) { return a_dispersive; }
        return *a_dispersive | *b_dispersive;
    }

public:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wo, Expr<float3> wi,
                                               TransportMode mode) const noexcept override {
        auto &&ctx = context<Context>();

        auto eval_a = a()->evaluate(wo, wi, mode);
        auto eval_b = b()->evaluate(wo, wi, mode);

        return _mix(eval_a, eval_b, ctx.ratio);
    }
    [[nodiscard]] Surface::Sample sample(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u,
                                         TransportMode mode) const noexcept override {
        auto &&ctx = context<Context>();

        auto sample = Surface::Sample::zero(swl().dimension());
        $if(u_lobe < ctx.ratio) {// sample a
            auto sample_a = a()->sample(wo, u_lobe / ctx.ratio, u, mode);
            auto eval_b = b()->evaluate(wo, sample_a.wi, mode);
            sample.eval = _mix(sample_a.eval, eval_b, ctx.ratio);
            sample.wi = sample_a.wi;
            sample.event = sample_a.event;
        }
        $else {// sample b
            auto sample_b = a()->sample(wo, (u_lobe - ctx.ratio) / (1.f - ctx.ratio), u, mode);
            auto eval_a = b()->evaluate(wo, sample_b.wi, mode);
            sample.eval = _mix(eval_a, sample_b.eval, ctx.ratio);
            sample.wi = sample_b.wi;
            sample.event = sample_b.event;
        };
        return sample;
    }
};

luisa::unique_ptr<Surface::Closure> MixSurfaceInstance::create_closure(const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto a = _a->create_closure(swl, time);
    auto b = _b->create_closure(swl, time);
    return luisa::make_unique<MixSurfaceClosure>(pipeline(), swl, time, std::move(a), std::move(b));
}

void MixSurfaceInstance::populate_closure(Surface::Closure *closure_in, const Interaction &it,
                                           Expr<float3> wo, Expr<float> eta_i) const noexcept {
    auto closure = static_cast<MixSurfaceClosure *>(closure_in);
    auto &swl = closure->swl();
    auto time = closure->time();
    auto ratio = _ratio == nullptr ? 0.5f : clamp(_ratio->evaluate(it, swl, time).x, 0.f, 1.f);

    MixSurfaceClosure::Context ctx{
        .it = it,
        .ratio = ratio};
    closure->bind(std::move(ctx));

    _a->populate_closure(closure->a(), it, wo, eta_i);
    _b->populate_closure(closure->b(), it, wo, eta_i);
}

using NormalMapMixSurface = NormalMapWrapper<
    MixSurface, MixSurfaceInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapMixSurface)

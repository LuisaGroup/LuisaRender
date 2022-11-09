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
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MixSurface::_build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
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
    Float _ratio;

private:
    [[nodiscard]] auto _mix(const Surface::Evaluation &eval_a,
                            const Surface::Evaluation &eval_b) const noexcept {
        auto t = 1.f - _ratio;
        return Surface::Evaluation{
            .f = lerp(eval_a.f, eval_b.f, t),
            .pdf = lerp(eval_a.pdf, eval_b.pdf, t)};
    }

public:
    MixSurfaceClosure(
        const MixSurfaceInstance *instance, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time, Expr<float> ratio,
        luisa::unique_ptr<Surface::Closure> a, luisa::unique_ptr<Surface::Closure> b) noexcept
        : Surface::Closure{instance, it, swl, time},
          _a{std::move(a)}, _b{std::move(b)}, _ratio{ratio} {
        LUISA_ASSERT(_a != nullptr || _b != nullptr,
                     "Creating closure for null MixSurface.");
    }
    [[nodiscard]] SampledSpectrum albedo() const noexcept override {
        auto albedo_a = _a->albedo();
        auto albedo_b = _b->albedo();
        return albedo_a * _ratio + albedo_b * (1.f - _ratio);
    }
    [[nodiscard]] Float2 roughness() const noexcept override {
        return _a->roughness() * _ratio + _b->roughness() * (1.f - _ratio);
    }

    [[nodiscard]] luisa::optional<Float> _opacity() const noexcept override {
        auto opacity_a = _a->opacity();
        auto opacity_b = _b->opacity();
        if (!opacity_a && !opacity_b) { return luisa::nullopt; }
        return lerp(opacity_b.value_or(1.f), opacity_a.value_or(1.f), _ratio);
    }
    [[nodiscard]] luisa::optional<Float> _eta() const noexcept override {
        auto eta_a = _a->eta();
        auto eta_b = _b->eta();
        if (!eta_a) { return eta_b; }
        if (!eta_b) { return eta_a; }
        return lerp(*eta_b, *eta_a, _ratio);
    }
    [[nodiscard]] luisa::optional<Bool> _is_dispersive() const noexcept override {
        auto a_dispersive = _a->is_dispersive();
        auto b_dispersive = _b->is_dispersive();
        if (!a_dispersive) { return b_dispersive; }
        if (!b_dispersive) { return a_dispersive; }
        return *a_dispersive | *b_dispersive;
    }

private:
    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        auto eval_a = _a->evaluate(wo, wi, mode);
        auto eval_b = _b->evaluate(wo, wi, mode);
        return _mix(eval_a, eval_b);
    }
    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u,
                                          TransportMode mode) const noexcept override {
        auto sample = Surface::Sample::zero(_swl.dimension());
        $if(u_lobe < _ratio) {// sample a
            auto sample_a = _a->sample(wo, u_lobe / _ratio, u, mode);
            auto eval_b = _b->evaluate(wo, sample_a.wi, mode);
            sample.eval = _mix(sample_a.eval, eval_b);
            sample.wi = sample_a.wi;
            sample.event = sample_a.event;
        }
        $else {// sample b
            auto sample_b = _b->sample(wo, (u_lobe - _ratio) / (1.f - _ratio), u, mode);
            auto eval_a = _a->evaluate(wo, sample_b.wi, mode);
            sample.eval = _mix(eval_a, sample_b.eval);
            sample.wi = sample_b.wi;
            sample.event = sample_b.event;
        };
        return sample;
    }
};

luisa::unique_ptr<Surface::Closure> MixSurfaceInstance::closure(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> eta_i, Expr<float> time) const noexcept {
    auto ratio = _ratio == nullptr ? 0.5f : clamp(_ratio->evaluate(it, swl, time).x, 0.f, 1.f);
    auto a = _a->closure(it, swl, eta_i, time);
    auto b = _b->closure(it, swl, eta_i, time);
    return luisa::make_unique<MixSurfaceClosure>(
        this, it, swl, time, ratio, std::move(a), std::move(b));
}

using NormalMapMixSurface = NormalMapWrapper<
    MixSurface, MixSurfaceInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapMixSurface)

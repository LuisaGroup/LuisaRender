//
// Created by Mike Smith on 2022/3/27.
//

#include <base/surface.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class MixSurface final : public Surface {

private:
    const Surface *_a;
    const Surface *_b;
    const Texture *_ratio;

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

public:
    MixSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _a{scene->load_surface(desc->property_node_or_default("a"))},
          _b{scene->load_surface(desc->property_node_or_default("b"))},
          _ratio{scene->load_texture(desc->property_node_or_default("ratio"))} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_null() const noexcept override {
        return (_a == nullptr || _a->is_null()) &&
               (_b == nullptr || _b->is_null());
    }
};

class MixSurfaceInstance final : public Surface::Instance {

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
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> time) const noexcept override;
    [[nodiscard]] auto ratio() const noexcept { return _ratio; }
};

luisa::unique_ptr<Surface::Instance> MixSurface::_build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    LUISA_ASSERT(!is_null(), "Building null MixSurface.");
    luisa::unique_ptr<Surface::Instance> a;
    luisa::unique_ptr<Surface::Instance> b;
    if (_a != nullptr && !_a->is_null()) [[likely]] {
        a = _a->build(pipeline, command_buffer);
    }
    if (_b != nullptr && !_b->is_null()) [[likely]] {
        b = _b->build(pipeline, command_buffer);
    }
    auto ratio = pipeline.build_texture(command_buffer, _ratio);
    return luisa::make_unique<MixSurfaceInstance>(
        pipeline, this, ratio, std::move(a), std::move(b));
}

class MixSurfaceClosure final : public Surface::Closure {

private:
    luisa::unique_ptr<Surface::Closure> _a;
    luisa::unique_ptr<Surface::Closure> _b;
    Float _ratio;

private:
    [[nodiscard]] auto _mix(const Surface::Evaluation &eval_a,
                            const Surface::Evaluation &eval_b) const noexcept {
        return Surface::Evaluation{
            .f = _ratio * eval_a.f + (1.f - _ratio) * eval_b.f,
             .pdf = lerp(eval_a.pdf, eval_b.pdf, 1.f - _ratio),
             .alpha = lerp(eval_a.alpha, eval_b.alpha, 1.f - _ratio),
             .eta = _ratio * eval_a.eta + (1.f - _ratio) * eval_b.eta};
    }

public:
    MixSurfaceClosure(
        const MixSurfaceInstance *instance, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time, Expr<float> ratio,
        luisa::unique_ptr<Surface::Closure> a, luisa::unique_ptr<Surface::Closure> b) noexcept
        : Surface::Closure{instance, it, swl, time},
          _a{std::move(a)}, _b{std::move(b)}, _ratio{ratio} {
        LUISA_ASSERT(_a != nullptr || _b == nullptr,
                     "Creating closure for null MixSurface.");
    }
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        if (_a == nullptr) [[unlikely]] {
            auto eval = _b->evaluate(wi);
            eval.f *= 1.f - _ratio;
            return eval;
        }
        if (_b == nullptr) [[unlikely]] {
            auto eval = _a->evaluate(wi);
            eval.f *= _ratio;
            return eval;
        }
        auto eval_a = _a->evaluate(wi);
        auto eval_b = _b->evaluate(wi);
        return _mix(eval_a, eval_b);
    }
    [[nodiscard]] Surface::Sample sample(Expr<float> u_lobe, Expr<float2> u) const noexcept override {
        if (_a == nullptr) [[unlikely]] {
            auto sample = _b->sample(u_lobe, u);
            sample.eval.f *= 1.f - _ratio;
            return sample;
        }
        if (_b == nullptr) [[unlikely]] {
            auto sample = _a->sample(u_lobe, u);
            sample.eval.f *= _ratio;
            return sample;
        }
        Surface::Sample sample{
            .wi = make_float3(0.f, 0.f, 1.f),
            .eval = {.f = SampledSpectrum{_swl.dimension()},
                     .pdf = 0.f,
                     .alpha = make_float2(),
                     .eta = SampledSpectrum{_swl.dimension(), 1.f}}};
        $if(u_lobe < _ratio) {// sample a
            auto sample_a = _a->sample(u_lobe / _ratio, u);
            auto eval_b = _b->evaluate(sample_a.wi);
            sample.wi = sample_a.wi;
            sample.eval = _mix(sample_a.eval, eval_b);
        }
        $else {// sample b
            auto sample_b = _b->sample((u_lobe - _ratio) / (1.f - _ratio), u);
            auto eval_a = _a->evaluate(sample_b.wi);
            sample.wi = sample_b.wi;
            sample.eval = _mix(eval_a, sample_b.eval);
        };
        return sample;
    }
    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        if (_a != nullptr) [[likely]] { _a->backward(wi, _ratio * df); }
        if (_b != nullptr) [[likely]] { _b->backward(wi, (1.f - _ratio) * df); }
        if (_a != nullptr && _b != nullptr) [[likely]] {
            if (auto ratio = instance<MixSurfaceInstance>()->ratio();
                ratio != nullptr && ratio->node()->requires_gradients()) {
                auto fa = _a->evaluate(wi).f;
                auto fb = _b->evaluate(wi).f;
                auto dt = ((fa - fb) * df).sum();// inner product
                ratio->backward(_it, _time, make_float4(dt, make_float3()));
            }
        }
    }
};

luisa::unique_ptr<Surface::Closure> MixSurfaceInstance::closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto ratio = _ratio == nullptr ? 0.5f : clamp(_ratio->evaluate(it, time).x, 0.f, 1.f);
    auto a = _a == nullptr ? nullptr : _a->closure(it, swl, time);
    auto b = _b == nullptr ? nullptr : _b->closure(it, swl, time);
    return luisa::make_unique<MixSurfaceClosure>(
        this, it, swl, time, ratio, std::move(a), std::move(b));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MixSurface)

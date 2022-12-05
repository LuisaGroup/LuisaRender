//
// Created by Mike Smith on 2022/4/9.
//

#include <numbers>

#include <util/sampling.h>
#include <util/imageio.h>
#include <base/environment.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

class Combined final : public Environment {

private:
    const Environment *_a;
    const Environment *_b;
    float2 _scales;

public:
    Combined(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc},
          _a{scene->load_environment(desc->property_node_or_default("a"))},
          _b{scene->load_environment(desc->property_node_or_default("b"))} {
        auto scale_a = std::max(desc->property_float_or_default("scale_a", 1.f), 0.f);
        auto scale_b = std::max(desc->property_float_or_default("scale_b", 1.f), 0.f);
        _scales = make_float2(scale_a, scale_b);
        if (_a == nullptr || _a->is_black()) [[unlikely]] { _scales.x = 0.f; }
        if (_b == nullptr || _b->is_black()) [[unlikely]] { _scales.y = 0.f; }
    }
    [[nodiscard]] auto scales() const noexcept { return _scales; }
    [[nodiscard]] bool is_black() const noexcept override { return all(_scales == 0.f); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class CombinedInstance final : public Environment::Instance {

private:
    luisa::unique_ptr<Instance> _a;
    luisa::unique_ptr<Instance> _b;

public:
    CombinedInstance(Pipeline &pipeline, const Environment *env,
                     luisa::unique_ptr<Instance> a, luisa::unique_ptr<Instance> b) noexcept
        : Environment::Instance{pipeline, env}, _a{std::move(a)}, _b{std::move(b)} {}

    [[nodiscard]] Environment::Evaluation evaluate(Expr<float3> wi,
                                                   const SampledWavelengths &swl,
                                                   Expr<float> time) const noexcept override {
        auto scales = node<Combined>()->scales();
        auto world_to_env = transpose(transform_to_world());
        auto wi_local = normalize(world_to_env * wi);
        if (_a != nullptr && _b != nullptr) [[likely]] {
            auto a_eval = _a->evaluate(wi_local, swl, time);
            auto b_eval = _b->evaluate(wi_local, swl, time);
            auto L = a_eval.L * scales.x + b_eval.L * scales.y;
            auto t = scales.y / (scales.x + scales.y);
            return {.L = a_eval.L * scales.x + b_eval.L * scales.y,
                    .pdf = lerp(a_eval.pdf, b_eval.pdf, t)};
        }
        if (_a != nullptr) [[unlikely]] {
            auto eval = _a->evaluate(wi_local, swl, time);
            return {.L = eval.L * scales.x, .pdf = eval.pdf};
        }
        auto eval = _b->evaluate(wi_local, swl, time);
        return {.L = eval.L * scales.y, .pdf = eval.pdf};
    }

    [[nodiscard]] Environment::Sample sample(const SampledWavelengths &swl,
                                             Expr<float> time,
                                             Expr<float2> u_in) const noexcept override {
        auto u = make_float2(u_in);
        auto scales = node<Combined>()->scales();
        auto sample = Environment::Sample::zero(swl.dimension());
        if (_a != nullptr && _b != nullptr) [[likely]] {
            auto weight_a = scales.x / (scales.x + scales.y);
            $if(u.x < weight_a) {// sample a
                u.x = u.x / weight_a;
                sample = _a->sample(swl, time, u);
                auto eval_b = _b->evaluate(sample.wi, swl, time);
                sample.eval.L = sample.eval.L * scales.x + eval_b.L * scales.y;
                sample.eval.pdf = lerp(sample.eval.pdf, eval_b.pdf, 1.f - weight_a);
            }
            $else {// sample b
                u.x = (u.x - weight_a) / (1.f - weight_a);
                sample = _b->sample(swl, time, u);
                auto eval_a = _a->evaluate(sample.wi, swl, time);
                sample.eval.L = eval_a.L * scales.x + sample.eval.L * scales.y;
                sample.eval.pdf = lerp(eval_a.pdf, sample.eval.pdf, 1.f - weight_a);
            };
        } else if (_a != nullptr) [[unlikely]] {
            sample = _a->sample(swl, time, u);
            sample.eval.L *= scales.x;
        } else {
            sample = _b->sample(swl, time, u);
            sample.eval.L *= scales.y;
        }
        sample.wi = normalize(transform_to_world() * sample.wi);
        return sample;
    }
};

luisa::unique_ptr<Environment::Instance> Combined::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    LUISA_ASSERT(any(_scales > 0.f), "Invalid scales in CombinedEnvironment.");
    auto a = _scales.x == 0.f ? nullptr : _a->build(pipeline, command_buffer);
    auto b = _scales.y == 0.f ? nullptr : _b->build(pipeline, command_buffer);
    return luisa::make_unique<CombinedInstance>(pipeline, this, std::move(a), std::move(b));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Combined)

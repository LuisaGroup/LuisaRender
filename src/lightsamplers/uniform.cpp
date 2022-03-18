//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <base/light_sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class UniformLightSampler final : public LightSampler {

public:
    luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    UniformLightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : LightSampler{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

class UniformLightSamplerInstance final : public LightSampler::Instance {

private:
    uint _light_buffer_id{0u};
    float _env_prob{0.f};

public:
    UniformLightSamplerInstance(const LightSampler *sampler, Pipeline &pipeline, CommandBuffer &command_buffer) noexcept
        : LightSampler::Instance{pipeline, sampler} {
        if (!pipeline.lights().empty()) {
            auto [view, buffer_id] = pipeline.arena_buffer<Light::Handle>(pipeline.lights().size());
            _light_buffer_id = buffer_id;
            command_buffer << view.copy_from(pipeline.instanced_lights().data())
                           << compute::commit();
        }
        if (auto env = pipeline.environment()) {
            auto n = static_cast<float>(pipeline.lights().size());
            _env_prob = env->node()->importance() /
                        (n + env->node()->importance());
        }
    }
    void update(CommandBuffer &, float) noexcept override {}
    [[nodiscard]] Light::Evaluation evaluate_hit(
        const Interaction &it, Expr<float3> p_from,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        if (pipeline().lights().empty()) [[unlikely]] {// no lights
            LUISA_WARNING_WITH_LOCATION("No lights in scene.");
            return {.L = make_float4(0.f), .pdf = 0.f};
        }
        Light::Evaluation eval;
        pipeline().dynamic_dispatch_light(it.shape()->light_tag(), [&](auto light) noexcept {
            auto closure = light->closure(swl, time);
            eval = closure->evaluate(it, p_from);
        });
        auto n = static_cast<float>(pipeline().lights().size());
        eval.pdf *= (1.f - _env_prob) / n;
        return eval;
    }
    [[nodiscard]] Light::Evaluation evaluate_miss(
        Expr<float3> wi, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        if (_env_prob == 0.f) [[unlikely]] {// no environment
            LUISA_WARNING_WITH_LOCATION("No environment in scene");
            return {.L = make_float4(0.f), .pdf = 0.f};
        }
        auto eval = pipeline().environment()->evaluate(wi, env_to_world, swl, time);
        eval.pdf *= _env_prob;
        return eval;
    }
    [[nodiscard]] Light::Sample sample(
        Sampler::Instance &sampler, const Interaction &it_from, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        if (_env_prob > 0.f) {// consider environment
            auto u = sampler.generate_1d();
            Light::Sample sample;
            $if(u < _env_prob) {
                sample = pipeline().environment()->sample(
                    sampler, it_from.p(), env_to_world, swl, time);
                sample.eval.pdf *= _env_prob;
            }
            $else {
                if (!pipeline().lights().empty()) {
                    auto n = static_cast<float>(pipeline().lights().size());
                    auto u_remapped = (u - _env_prob) / (1.f - _env_prob);
                    auto i = cast<uint>(clamp(u_remapped * n, 0.f, n - 1.f));
                    auto handle = pipeline().buffer<Light::Handle>(_light_buffer_id).read(i);
                    pipeline().dynamic_dispatch_light(handle.light_tag, [&](auto light) noexcept {
                        auto closure = light->closure(swl, time);
                        sample = closure->sample(sampler, handle.instance_id, it_from.p());
                    });
                    sample.eval.pdf *= (1.f - _env_prob) / n;
                }
            };
            return sample;
        }
        if (!pipeline().lights().empty()) {
            auto u = sampler.generate_1d();
            auto n = static_cast<float>(pipeline().lights().size());
            auto i = cast<uint>(clamp(u * n, 0.f, n - 1.f));
            auto handle = pipeline().buffer<Light::Handle>(_light_buffer_id).read(i);
            Light::Sample sample;
            pipeline().dynamic_dispatch_light(handle.light_tag, [&](auto light) noexcept {
                auto closure = light->closure(swl, time);
                sample = closure->sample(sampler, handle.instance_id, it_from.p());
            });
            sample.eval.pdf *= 1.f / n;
            return sample;
        }
        LUISA_WARNING_WITH_LOCATION("No light or environment to sample.");
        return {.eval = {.L = make_float4(), .pdf = 0.f},
                .wi = make_float3(0.f, 1.f, 0.f),
                .distance = std::numeric_limits<float>::max()};
    }
};

unique_ptr<LightSampler::Instance> UniformLightSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<UniformLightSamplerInstance>(
        this, pipeline, command_buffer);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::UniformLightSampler)

//
// Created by Mike Smith on 2022/1/14.
//

#include <luisa-compute.h>
#include <util/sampling.h>
#include <scene/interaction.h>
#include <scene/environment.h>

namespace luisa::render {

class ConstantEnvironment final : public Environment {

private:
    float3 _emission;

public:
    ConstantEnvironment(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc} {
        auto emission = desc->property_float3_or_default(
            "emission", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default("emission", 1.0f));
            }));
        auto scale = desc->property_float_or_default("scale", 1.0f);
        _emission = max(emission * scale, 0.0f);
    }
    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] bool is_black() const noexcept override { return all(_emission == 0.0f); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "constant"; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ConstantEnvironmentInstance final : public Environment::Instance {

private:
    float3 _emission;

public:
    ConstantEnvironmentInstance(Pipeline &ppl, const ConstantEnvironment *env) noexcept
        : Environment::Instance{ppl, env}, _emission{env->emission()} {}
    [[nodiscard]] auto evaluate() const noexcept {
        return Light::Evaluation{.L = _emission, .pdf = uniform_sphere_pdf()};
    }
    [[nodiscard]] Light::Evaluation evaluate(Expr<float3>, Expr<float3x3>, Expr<float>) const noexcept override { return evaluate(); }
    [[nodiscard]] Light::Sample sample(Sampler::Instance &sampler, const Interaction &it_from, Expr<float3x3> env_to_world, Expr<float>) const noexcept override {
        auto wi = sample_uniform_sphere(sampler.generate_2d());
        return {.eval = evaluate(), .shadow_ray = it_from.spawn_ray(env_to_world * wi)};
    }
};

luisa::unique_ptr<Environment::Instance> ConstantEnvironment::build(Pipeline &pipeline, CommandBuffer &) const noexcept {
    return luisa::make_unique<ConstantEnvironmentInstance>(pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantEnvironment)

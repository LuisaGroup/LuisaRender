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
    float3 _rsp;
    float _scale{0.0f};

public:
    ConstantEnvironment(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc} {
        auto emission = desc->property_float3_or_default(
            "emission", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default("emission", 1.0f));
            }));
        auto scale = desc->property_float_or_default("scale", 1.0f);
        std::tie(_rsp, _scale) = RGB2SpectrumTable::srgb().decode_unbound(max(emission * scale, 0.0f));
    }
    [[nodiscard]] auto rsp() const noexcept { return _rsp; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "constant"; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

struct ConstantEnvironmentInstance final : public Environment::Instance {
    ConstantEnvironmentInstance(Pipeline &ppl, const ConstantEnvironment *env) noexcept
        : Environment::Instance{ppl, env} {}
    [[nodiscard]] auto evaluate(const SampledWavelengths &swl) const noexcept {
        auto env = static_cast<const ConstantEnvironment *>(node());
        RGBIlluminantSpectrum spec{
            RGBSigmoidPolynomial{env->rsp()}, env->scale(),
            DenselySampledSpectrum::cie_illum_d6500()};
        return Light::Evaluation{.L = spec.sample(swl), .pdf = uniform_sphere_pdf()};
    }
    [[nodiscard]] Light::Evaluation evaluate(
        Expr<float3>, Expr<float3x3>,
        const SampledWavelengths &swl, Expr<float>) const noexcept override { return evaluate(swl); }
    [[nodiscard]] Light::Sample sample(
        Sampler::Instance &sampler, const Interaction &it_from, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float>) const noexcept override {
        auto wi = sample_uniform_sphere(sampler.generate_2d());
        return {.eval = evaluate(swl), .shadow_ray = it_from.spawn_ray(env_to_world * wi)};
    }
};

luisa::unique_ptr<Environment::Instance> ConstantEnvironment::build(Pipeline &pipeline, CommandBuffer &) const noexcept {
    return luisa::make_unique<ConstantEnvironmentInstance>(pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConstantEnvironment)

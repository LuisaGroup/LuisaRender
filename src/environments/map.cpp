//
// Created by Mike Smith on 2022/1/15.
//

#include <util/sampling.h>
#include <util/imageio.h>
#include <base/interaction.h>
#include <base/environment.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class EnvironmentMapping final : public Environment {

private:
    const Texture *_emission;
    float _scale;

public:
    EnvironmentMapping(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc},
          _emission{scene->load_texture(desc->property_node_or_default(
              "emission", SceneNodeDesc::shared_default_texture("ConstIllum")))},
          _scale{std::max(desc->property_float_or_default("scale", 1.0f), 0.0f)} {
        if (_emission->category() != Texture::Category::ILLUMINANT) [[unlikely]] {
            LUISA_ERROR(
                "Non-illuminant textures are not "
                "allowed in environment mapping. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f || _emission->is_black(); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

using namespace luisa::compute;

class EnvironmentMappingInstance final : public Environment::Instance {

private:
    const Texture::Instance *_texture;

private:
    [[nodiscard]] auto _evaluate(Expr<float3> wi_local, const SampledWavelengths &swl, Expr<float> time) const noexcept {
        auto env = node<EnvironmentMapping>();
        auto theta = acos(wi_local.y);
        auto phi = atan2(wi_local.x, wi_local.z);
        auto u = -0.5f * inv_pi * phi;
        auto v = theta * inv_pi;
        Interaction it{-wi_local, make_float2(u, v)};
        auto L = _texture->evaluate(it, swl, time);
        return Light::Evaluation{.L = L * env->scale(), .pdf = uniform_sphere_pdf()};
    }

public:
    EnvironmentMappingInstance(const Pipeline &pipeline, const Environment *env, const Texture::Instance *texture) noexcept
        : Environment::Instance{pipeline, env}, _texture{texture} {}
    // TODO: importance sampling
    [[nodiscard]] Light::Evaluation evaluate(
        Expr<float3> wi, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto world_to_env = transpose(env_to_world);
        return _evaluate(world_to_env * wi, swl, time);
    }
    [[nodiscard]] Light::Sample sample(
        Sampler::Instance &sampler, const Interaction &it_from, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto local_wi = sample_uniform_sphere(sampler.generate_2d());
        return {.eval = _evaluate(local_wi, swl, time),
                .shadow_ray = it_from.spawn_ray(env_to_world * local_wi)};
    }
};

unique_ptr<Environment::Instance> EnvironmentMapping::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto texture = pipeline.build_texture(command_buffer, _emission);
    return luisa::make_unique<EnvironmentMappingInstance>(pipeline, this, texture);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::EnvironmentMapping)

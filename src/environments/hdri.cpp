//
// Created by Mike Smith on 2022/1/15.
//

#include <luisa-compute.h>
#include <util/sampling.h>
#include <util/imageio.h>
#include <base/interaction.h>
#include <base/environment.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

[[nodiscard]] static auto default_emission_texture_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{
            "__hdri_environment_default_emission_texture",
            SceneNodeTag::TEXTURE};
        d.define(SceneNodeTag::TEXTURE, "constillum", {});
        return &d;
    }();
    return desc;
}

class HDRIEnvironment final : public Environment {

private:
    const Texture *_emission;
    float _scale;

public:
    HDRIEnvironment(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc},
          _emission{scene->load_texture(desc->property_node_or_default(
              "emission", default_emission_texture_desc()))},
          _scale{std::max(desc->property_float_or_default("scale", 1.0f), 0.0f)} {
        if (!_emission->is_illuminant()) [[unlikely]] {
            LUISA_ERROR(
                "Non-illuminant textures are not "
                "allowed in HDRI environments. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f || _emission->is_black(); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "hdri"; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

using namespace luisa::compute;

class HDRIEnvironmentInstance final : public Environment::Instance {

private:
    TextureHandle _texture;

private:
    [[nodiscard]] auto _evaluate(Expr<float3> wi_local, const SampledWavelengths &swl, Expr<float> time) const noexcept {
        auto env = static_cast<const HDRIEnvironment *>(node());
        auto handle = def<TextureHandle>();
        handle.id_and_tag = _texture.id_and_tag;
        for (auto i = 0u; i < std::size(_texture.compressed_v); i++) {
            handle.compressed_v[i] = _texture.compressed_v[i];
        }
        auto L = env->emission()->evaluate(pipeline(), handle, wi_local, swl, time);
        return Light::Evaluation{.L = L * env->scale(), .pdf = uniform_sphere_pdf()};
    }

public:
    HDRIEnvironmentInstance(Pipeline &pipeline, CommandBuffer &command_buffer, const HDRIEnvironment *env) noexcept
        : Environment::Instance{pipeline, env},
          _texture{*pipeline.encode_texture(env->emission(), command_buffer)} {}
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

unique_ptr<Environment::Instance> HDRIEnvironment::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<HDRIEnvironmentInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HDRIEnvironment)

//
// Created by Mike Smith on 2022/4/9.
//

#include <numbers>

#include <util/sampling.h>
#include <util/imageio.h>
#include <base/interaction.h>
#include <base/environment.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

class Directional final : public Environment {

private:
    const Texture *_emission;
    float _scale;
    float _cos_half_angle{};
    float _direction[3]{};
    bool _visible;

public:
    Directional(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc},
          _emission{scene->load_texture(desc->property_node_or_default(
              "emission", SceneNodeDesc::shared_default_texture("Constant")))},
          _scale{std::max(desc->property_float_or_default("scale", 1.0f), 0.0f)},
          _visible{desc->property_bool_or_default("visible", true)} {
        if (!_emission->is_constant()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Directional environment emission is not constant. "
                "This may lead to unexpected results.");
        }
        LUISA_RENDER_PARAM_CHANNEL_CHECK(Directional, emission, >=, 3);
        auto angle = std::clamp(desc->property_float_or_default("angle", 1.0f), 1e-3f, 360.0f);
        auto cos_half_angle = std::cos(.5 * angle * std::numbers::pi / 180.0);
        _cos_half_angle = static_cast<float>(cos_half_angle);
        if (desc->property_bool_or_default("normalize", true)) {
            _scale = static_cast<float>(2. * _scale / (1. - cos_half_angle));
        }
        auto direction = normalize(desc->property_float3_or_default("direction", float3{0.0f, 1.0f, 0.0f}));
        _direction[0] = direction[0];
        _direction[1] = direction[1];
        _direction[2] = direction[2];
    }
    [[nodiscard]] auto visible() const noexcept { return _visible; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto direction() const noexcept { return make_float3(_direction[0], _direction[1], _direction[2]); }
    [[nodiscard]] auto cos_half_angle() const noexcept { return _cos_half_angle; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f || _emission->is_black(); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class DirectionalInstance final : public Environment::Instance {

private:
    const Texture::Instance *_texture;

private:
    [[nodiscard]] auto _evaluate(Expr<float3> wi_local, const SampledWavelengths &swl, Expr<float> time) const noexcept {
        auto env = node<Directional>();
        Interaction it{-wi_local, make_float2(.5f)};
        auto L = _texture->evaluate_illuminant_spectrum(it, swl, time);
        auto pdf = uniform_cone_pdf(env->cos_half_angle());
        auto valid = env->cos_half_angle() < abs_cos_theta(wi_local);
        return Light::Evaluation{.L = L * ite(valid, env->scale(), 0.f),
                                 .pdf = ite(valid, pdf, 0.f)};
    }

public:
    DirectionalInstance(Pipeline &pipeline, const Environment *env, const Texture::Instance *texture) noexcept
        : Environment::Instance{pipeline, env}, _texture{texture} {}
    [[nodiscard]] Light::Evaluation evaluate(
        Expr<float3> wi, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto env = node<Directional>();
        if (!env->visible()) { return Light::Evaluation::zero(swl.dimension()); }
        auto world_to_env = transpose(transform_to_world());
        auto frame = Frame::make(env->direction());
        auto wi_local = normalize(frame.world_to_local(world_to_env * wi));
        return _evaluate(wi_local, swl, time);
    }
    [[nodiscard]] Light::Sample sample(
        Expr<float3> p_from, const SampledWavelengths &swl,
        Expr<float> time, Expr<float2> u) const noexcept override {
        auto env = node<Directional>();
        auto wi_local = sample_uniform_cone(u, env->cos_half_angle());
        auto frame = Frame::make(env->direction());
        return {.eval = _evaluate(wi_local, swl, time),
                .wi = normalize(transform_to_world() * frame.local_to_world(wi_local)),
                .distance = std::numeric_limits<float>::max()};
    }
};

luisa::unique_ptr<Environment::Instance> Directional::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto texture = pipeline.build_texture(command_buffer, _emission);
    return luisa::make_unique<DirectionalInstance>(pipeline, this, texture);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Directional)

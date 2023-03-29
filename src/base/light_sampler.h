//
// Created by Mike Smith on 2022/1/10.
//

#pragma once

#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/light.h>
#include <base/environment.h>

namespace luisa::render {

class Interaction;

using compute::Float;
using compute::UInt;

class LightSampler : public SceneNode {

public:
    struct Selection {
        UInt tag;
        Float prob;
    };
    static constexpr auto selection_environment = ~0u;

    using Evaluation = Light::Evaluation;
    struct Sample {
        Evaluation eval;
        Var<Ray> shadow_ray;
        [[nodiscard]] static Sample zero(uint spec_dim) noexcept;
        [[nodiscard]] static Sample from_light(const Light::Sample &s,
                                               const Interaction &it_from) noexcept;
        [[nodiscard]] static Sample from_environment(const Environment::Sample &s,
                                                     const Interaction &it_from) noexcept;
    };

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const LightSampler *_sampler;

    private:
        [[nodiscard]] virtual Light::Sample _sample_light(const Interaction &it_from,
                                                          Expr<uint> tag, Expr<float2> u,
                                                          const SampledWavelengths &swl,
                                                          Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Environment::Sample _sample_environment(Expr<float2> u,
                                                                      const SampledWavelengths &swl,
                                                                      Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual LightSampler::Sample _sample_light_le(
                                                          Expr<uint> tag, Expr<float2> u_light, Expr<float2> u_direction,
                                                          const SampledWavelengths &swl,
                                                          Expr<float> time) const noexcept = 0;

    public:
        explicit Instance(const Pipeline &pipeline, const LightSampler *light_dist) noexcept
            : _pipeline{pipeline}, _sampler{light_dist} {}
        virtual ~Instance() noexcept = default;

        template<typename T = LightSampler>
            requires std::is_base_of_v<LightSampler, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_sampler); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Evaluation evaluate_hit(
            const Interaction &it, Expr<float3> p_from,
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Evaluation evaluate_miss(
            Expr<float3> wi, const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Selection select(
            const Interaction &it_from, Expr<float> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Selection select(
            Expr<float> u, const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] Sample sample_light(
            const Interaction &it_from, const Selection &sel, Expr<float2> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] Sample sample_environment(
            const Interaction &it_from, const Selection &sel, Expr<float2> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] Sample sample_light_le(
            const Selection &sel, Expr<float2> u_light, Expr<float2> u_direction,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] Sample sample_environment_le(
            const Selection &sel, Expr<float2> u_light, Expr<float2> u_direction,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Sample sample_selection(
            const Interaction &it_from, const Selection &sel, Expr<float2> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Sample sample_selection_le(
            const Selection &sel, Expr<float2> u_light, Expr<float2> u_direction,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Sample sample(
            const Interaction &it_from, Expr<float> u_sel, Expr<float2> u_light,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Sample sample_le(
            Expr<float> u_sel, Expr<float2> u_light, Expr<float2> u_direction,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
    };

public:
    LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::LightSampler::Selection)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::LightSampler::Sample)

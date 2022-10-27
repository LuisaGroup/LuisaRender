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

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const LightSampler *_sampler;

    private:
        [[nodiscard]] virtual Light::Sample _sample_light(
            const Interaction &it_from, Expr<uint> tag, Expr<float2> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Light::Sample _sample_environment(
            const Interaction &it_from, Expr<float2> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;

    public:
        explicit Instance(const Pipeline &pipeline, const LightSampler *light_dist) noexcept
            : _pipeline{pipeline}, _sampler{light_dist} {}
        virtual ~Instance() noexcept = default;

        template<typename T = LightSampler>
            requires std::is_base_of_v<LightSampler, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_sampler); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Light::Evaluation evaluate_hit(
            const Interaction &it, Expr<float3> p_from,
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Light::Evaluation evaluate_miss(
            Expr<float3> wi, const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Selection select(
            const Interaction &it_from, Expr<float> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] Light::Sample sample_light(
            const Interaction &it_from, const Selection &sel, Expr<float2> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] Light::Sample sample_environment(
            const Interaction &it_from, Expr<float> prob, Expr<float2> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Light::Sample sample_selection(
            const Interaction &it_from, const Selection &sel, Expr<float2> u,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Light::Sample sample(
            const Interaction &it_from, Expr<float> u_sel, Expr<float2> u_light,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
    };

public:
    LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

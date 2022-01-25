//
// Created by Mike Smith on 2022/1/10.
//

#pragma once

#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/light.h>

namespace luisa::render {

class Interaction;

using compute::UInt;
using compute::Float;

class LightSampler : public SceneNode {

public:
    struct Selection {
        UInt instance_id;
        UInt light_tag;
        Float pmf;
        [[nodiscard]] auto is_environment() const noexcept {
            return instance_id == ~0u;
        }
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const LightSampler *_sampler;

    public:
        explicit Instance(Pipeline &pipeline, const LightSampler *light_dist) noexcept
            : _pipeline{pipeline}, _sampler{light_dist} {}
        virtual ~Instance() noexcept = default;
        [[nodiscard]] uint light_count() const noexcept;
        [[nodiscard]] auto node() const noexcept { return _sampler; }
        [[nodiscard]] const auto &pipeline() const noexcept { return _pipeline; }
        virtual void update(CommandBuffer &command_buffer, float time) noexcept = 0;
        [[nodiscard]] virtual Float pmf(
            const Interaction &it,
            const SampledWavelengths &swl) const noexcept = 0;
        [[nodiscard]] virtual Selection select(
            Sampler::Instance &sampler, const Interaction &it,
            const SampledWavelengths &swl) const noexcept = 0;
        [[nodiscard]] virtual Light::Evaluation evaluate(
            const Interaction &it, Expr<float3> p_from,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Light::Sample sample(
            Sampler::Instance &sampler, const Interaction &it_from,
            const SampledWavelengths &swl, Expr<float> time) const noexcept;
    };

public:
    LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

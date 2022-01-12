//
// Created by Mike Smith on 2022/1/10.
//

#pragma once

#include <scene/scene_node.h>
#include <scene/sampler.h>
#include <scene/light.h>

namespace luisa::render {

class Interaction;

using compute::UInt;
using compute::Float;

class LightSampler : public SceneNode {

public:
    struct Selection {
        UInt instance_id;
        UInt light_tag;
        Float pdf;
    };

    class Instance {

    private:
        Pipeline &_pipeline;
        const LightSampler *_sampler;

    public:
        explicit Instance(Pipeline &pipeline, const LightSampler *light_dist) noexcept
            : _pipeline{pipeline}, _sampler{light_dist} {}
        virtual ~Instance() noexcept = default;
        virtual void update(Stream &stream) noexcept = 0;
        [[nodiscard]] auto node() const noexcept { return _sampler; }
        [[nodiscard]] const auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Float pdf_selection(const Interaction &it) const noexcept = 0;
        [[nodiscard]] virtual Selection select(Sampler::Instance &sampler, const Interaction &it) const noexcept = 0;
        [[nodiscard]] virtual Light::Evaluation evaluate(const Interaction &it, Expr<float3> p_from) const noexcept;
        [[nodiscard]] virtual Light::Sample sample(Sampler::Instance &sampler, const Interaction &it_from) const noexcept;
    };

public:
    LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

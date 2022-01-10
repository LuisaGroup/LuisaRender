//
// Created by Mike Smith on 2022/1/10.
//

#pragma once

#include <scene/scene_node.h>
#include <scene/sampler.h>
#include <scene/light.h>

namespace luisa::render {

class Interaction;

class LightSampler : public SceneNode {

public:
    class Instance {

    private:
        const LightSampler *_sampler;

    public:
        explicit Instance(const LightSampler *light_dist) noexcept : _sampler{light_dist} {}
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto node() const noexcept { return _sampler; }
        [[nodiscard]] virtual Light::Evaluation evaluate(const Interaction &it) const noexcept = 0;
        [[nodiscard]] virtual Light::Sample sample(Sampler::Instance &sampler, const Interaction &it) const noexcept = 0;
    };

public:
    LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

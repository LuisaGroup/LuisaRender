//
// Created by Mike Smith on 2022/1/14.
//

#pragma once

#include <scene/scene_node.h>
#include <scene/light.h>

namespace luisa::render {

class Transform;

class Environment : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const Environment *_env;
        float _select_prob;

    public:
        explicit Instance(Pipeline &pipeline, const Environment *env) noexcept;
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto node() const noexcept { return _env; }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto selection_prob() const noexcept { return _select_prob; }
        [[nodiscard]] virtual Light::Evaluation evaluate(Expr<float3> wi, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Light::Sample sample(Sampler::Instance &sampler, const Interaction &it_from, Expr<float> time) const noexcept = 0;
    };

private:
    const Transform *_transform;
    float _importance;// importance w.r.t. an average light

public:
    Environment(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] auto importance() const noexcept { return _importance; }
    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}

//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <scene/scene_node.h>

namespace luisa::render {

class Sampler;

class Integrator : public SceneNode {

public:
    class Instance : public SceneNode::Instance {

    private:
        const Integrator *_integrator;

    public:
        explicit Instance(const Integrator *integrator) noexcept : _integrator{integrator} {}
        [[nodiscard]] auto integrator() const noexcept { return _integrator; }
        virtual void render(Stream &stream, Pipeline &pipeline) noexcept = 0;
    };

private:
    const Sampler *_sampler;

public:
    Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}

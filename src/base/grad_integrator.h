//
// Created by ChenXin on 2022/2/24.
//

#pragma once

#include "scene_node.h"

namespace luisa::render {

class Sampler;
class LightSampler;

class GradIntegrator : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const GradIntegrator *_grad_integrator;

    public:
        explicit Instance(const Pipeline &pipeline, const GradIntegrator *grad_integrator) noexcept
            : _pipeline{pipeline}, _grad_integrator{grad_integrator} {}
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto node() const noexcept { return _grad_integrator; }
        virtual void integrate(Stream &stream) noexcept = 0;
    };

private:
    const Sampler *_sampler;
    const LightSampler *_light_sampler;

public:
    GradIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}
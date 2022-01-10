//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <scene/scene_node.h>

namespace luisa::render {

class Sampler;
class LightSampler;

class Integrator : public SceneNode {

public:
    class Instance {

    private:
        const Integrator *_integrator;

    public:
        explicit Instance(const Integrator *integrator) noexcept : _integrator{integrator} {}
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto node() const noexcept { return _integrator; }
        virtual void render(Stream &stream) noexcept = 0;
    };

private:
    const Sampler *_sampler;
    const LightSampler *_light_sampler;

public:
    Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}

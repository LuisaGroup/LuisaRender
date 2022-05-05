//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <runtime/command_buffer.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/spectrum.h>
#include <base/light_sampler.h>
#include "loss.h"
#include "optimizer.h"

namespace luisa::render {

class Pipeline;
using compute::CommandBuffer;

class Integrator : public SceneNode {

public:
    class Instance {

    private:
        Pipeline &_pipeline;
        const Integrator *_integrator;
        luisa::unique_ptr<Sampler::Instance> _sampler;
        luisa::unique_ptr<LightSampler::Instance> _light_sampler;
        luisa::unique_ptr<Spectrum::Instance> _spectrum;

    public:
        explicit Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Integrator *integrator) noexcept;
        virtual ~Instance() noexcept = default;

        template<typename T = Integrator>
            requires std::is_base_of_v<Integrator, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_integrator); }
        [[nodiscard]] auto &pipeline() noexcept { return _pipeline; }
        [[nodiscard]] const auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto sampler() noexcept { return _sampler.get(); }
        [[nodiscard]] auto sampler() const noexcept { return _sampler.get(); }
        [[nodiscard]] auto light_sampler() noexcept { return _light_sampler.get(); }
        [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler.get(); }
        [[nodiscard]] auto spectrum() noexcept { return _spectrum.get(); }
        [[nodiscard]] auto spectrum() const noexcept { return _spectrum.get(); }
        virtual void render(Stream &stream) noexcept = 0;
    };

private:
    const Sampler *_sampler;
    const LightSampler *_light_sampler;
    const Spectrum *_spectrum;

public:
    Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto spectrum() const noexcept { return _spectrum; }
    [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler; }
    [[nodiscard]] virtual bool is_differentiable() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

class DifferentiableIntegrator : public Integrator {

public:
    class Instance : public Integrator::Instance {

    private:
        luisa::unique_ptr<Loss::Instance> _loss;

    public:
        explicit Instance(Pipeline &pipeline, CommandBuffer &command_buffer,
                          const DifferentiableIntegrator *integrator) noexcept;
        [[nodiscard]] auto loss() const noexcept { return _loss.get(); }
    };

private:
    Loss *_loss;
    Optimizer _optimizer;
    mutable float _learning_rate;
    uint _iterations;
    int _display_camera_index;
    bool _save_process;

public:
    DifferentiableIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept;

    [[nodiscard]] bool is_differentiable() const noexcept override { return true; }
    [[nodiscard]] auto loss() const noexcept { return _loss; }
    [[nodiscard]] auto optimizer() const noexcept { return _optimizer; }
    [[nodiscard]] float &learning_rate() const noexcept { return _learning_rate; }
    [[nodiscard]] auto iterations() const noexcept { return _iterations; }
    [[nodiscard]] int display_camera_index() const noexcept { return _display_camera_index; }
    [[nodiscard]] bool save_process() const noexcept { return _save_process; }
};

}// namespace luisa::render

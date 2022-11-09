//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <runtime/command_buffer.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/spectrum.h>
#include <base/light_sampler.h>
#include <base/camera.h>

namespace luisa::render {

class Pipeline;
class Display;
using compute::CommandBuffer;

class Integrator : public SceneNode {

public:
    class Instance {

    private:
        Pipeline &_pipeline;
        const Integrator *_integrator;
        luisa::unique_ptr<Sampler::Instance> _sampler;
        luisa::unique_ptr<LightSampler::Instance> _light_sampler;

    public:
        explicit Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Integrator *integrator) noexcept;
        virtual ~Instance() noexcept = default;

        template<typename T = Integrator>
            requires std::is_base_of_v<Integrator, T>
        [[nodiscard]] auto
        node() const noexcept { return static_cast<const T *>(_integrator); }
        [[nodiscard]] auto &pipeline() noexcept { return _pipeline; }
        [[nodiscard]] const auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto sampler() noexcept { return _sampler.get(); }
        [[nodiscard]] auto sampler() const noexcept { return _sampler.get(); }
        [[nodiscard]] auto light_sampler() noexcept { return _light_sampler.get(); }
        [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler.get(); }
        virtual void render(Stream &stream) noexcept = 0;
    };

private:
    const Sampler *_sampler;
    const LightSampler *_light_sampler;

public:
    Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

class ProgressiveIntegrator : public Integrator {

public:
    class Instance : public Integrator::Instance {

    private:
        luisa::unique_ptr<Display> _display;

    protected:
        [[nodiscard]] virtual Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                                        Expr<uint2> pixel_id, Expr<float> time) const noexcept;
        virtual void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept;

    public:
        Instance(Pipeline &pipeline,
                 CommandBuffer &command_buffer,
                 const ProgressiveIntegrator *node) noexcept;
        ~Instance() noexcept override;
        void render(Stream &stream) noexcept override;
        [[nodiscard]] auto display() noexcept { return _display.get(); }
    };

private:
    uint16_t _display_interval;
    bool _display;

public:
    ProgressiveIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto display_enabled() const noexcept { return _display; }
    [[nodiscard]] auto display_interval() const noexcept { return static_cast<uint>(_display_interval); }
};

}// namespace luisa::render

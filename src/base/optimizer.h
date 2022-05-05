//
// Created by ChenXin on 2022/4/11.
//

#pragma once

#include <luisa-compute.h>

#include <base/scene_node.h>

namespace luisa::render {

class Optimizer : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const Optimizer *_optimizer;

    public:
        Instance(const Pipeline &pipeline, const Optimizer *optimizer) noexcept
            : _pipeline{pipeline}, _optimizer{optimizer} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Optimizer>
            requires std::is_base_of_v<Optimizer, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_optimizer); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }

    public:
        virtual void clear_gradients(CommandBuffer &command_buffer) noexcept = 0;
        virtual void apply_gradients(CommandBuffer &command_buffer, float alpha) noexcept = 0;
        /// Apply then clear the gradients
        virtual void step(CommandBuffer &command_buffer, float alpha) noexcept {
            apply_gradients(command_buffer, alpha);
            clear_gradients(command_buffer);
        }
    };

private:
    float _learning_rate;

public:
    Optimizer(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

    [[nodiscard]] auto learning_rate() const noexcept { return _learning_rate; }
};

}// namespace luisa::render
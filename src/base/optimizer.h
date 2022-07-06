//
// Created by ChenXin on 2022/4/11.
//

#pragma once

#include <luisa-compute.h>

#include <base/scene_node.h>

namespace luisa::render {

using namespace luisa::compute;

class Optimizer : public SceneNode {

public:
    class Instance {

    protected:
        const Pipeline &_pipeline;
        const Optimizer *_optimizer;
        const uint _length;

    public:
        Instance(const Pipeline &pipeline, const Optimizer *optimizer, const uint length) noexcept
            : _pipeline{pipeline}, _optimizer{optimizer}, _length{length} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Optimizer>
            requires std::is_base_of_v<Optimizer, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_optimizer); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }

    public:
        virtual void apply_gradients(CommandBuffer &command_buffer, float alpha, BufferView<uint> grad) noexcept = 0;
    };

protected:
    float _learning_rate;

public:
    Optimizer(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer, uint length) const noexcept = 0;

    [[nodiscard]] auto learning_rate() const noexcept { return _learning_rate; }
};

}// namespace luisa::render
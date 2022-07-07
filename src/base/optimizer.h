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

    private:
        Pipeline &_pipeline;
        const Optimizer *_optimizer;

    protected:
        uint _length = -1u;
        Shader1D<Buffer<uint>> _clear_uint_buffer;
        Shader1D<Buffer<float>> _clear_float_buffer;

    public:
        explicit Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Optimizer *optimizer) noexcept;
        virtual ~Instance() noexcept = default;
        template<typename T = Optimizer>
            requires std::is_base_of_v<Optimizer, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_optimizer); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }

    public:
        // allocate buffer/... space
        virtual void initialize(CommandBuffer &command_buffer, uint length, BufferView<float> x0) noexcept;
        virtual void step(CommandBuffer &command_buffer, BufferView<float> xi, BufferView<uint> gradients) noexcept = 0;
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
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
        luisa::optional<BufferView<float2>> _ranges;
        luisa::optional<BufferView<float>> _xi;
        luisa::optional<BufferView<float>> _gradients;

        Shader1D<Buffer<uint>> _clear_uint_buffer;
        Shader1D<Buffer<float>> _clear_float_buffer;
        Shader1D<Buffer<float>, Buffer<float>, Buffer<float2>, float> _clamp_range;

    public:
        explicit Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Optimizer *optimizer) noexcept;
        virtual ~Instance() noexcept = default;
        template<typename T = Optimizer>
            requires std::is_base_of_v<Optimizer, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_optimizer); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }

    public:
        // allocate buffer/... space
        virtual void initialize(CommandBuffer &command_buffer, uint length, BufferView<float> xi, BufferView<float> gradients, BufferView<float2> ranges) noexcept;
        virtual void step(CommandBuffer &command_buffer) noexcept = 0;
        void clamp_range(CommandBuffer &command_buffer) noexcept;
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
//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <dsl/syntax.h>
#include <base/scene_node.h>

namespace luisa::render {

using compute::Expr;
using compute::Float;
using compute::Float2;

class Sampler : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const Sampler *_sampler;

    public:
        explicit Instance(const Pipeline &pipeline, const Sampler *sampler) noexcept
            : _pipeline{pipeline}, _sampler{sampler} {}
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto node() const noexcept { return _sampler; }

        // interfaces
        virtual void reset(CommandBuffer &command_buffer, uint2 resolution, uint spp) noexcept = 0;
        virtual void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept = 0;
        virtual void save_state() noexcept = 0;
        virtual void load_state(Expr<uint2> pixel) noexcept = 0;
        [[nodiscard]] virtual Float generate_1d() noexcept = 0;
        [[nodiscard]] virtual Float2 generate_2d() noexcept = 0;
    };

public:
    Sampler(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

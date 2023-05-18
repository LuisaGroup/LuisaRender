//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <dsl/syntax.h>
#include <util/command_buffer.h>
#include <base/scene_node.h>

namespace luisa::render {

using compute::Expr;
using compute::Float;
using compute::Float2;

class Sampler : public SceneNode {

private:
    uint _seed;

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

        template<typename T = Sampler>
            requires std::is_base_of_v<Sampler, T>
        [[nodiscard]] auto node() const noexcept {
            return static_cast<const T *>(_sampler);
        }

        // interfaces
        virtual void reset(CommandBuffer &command_buffer, uint2 resolution, uint state_count, uint spp) noexcept = 0;
        virtual void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept = 0;
        virtual void save_state(Expr<uint> state_id) noexcept = 0;
        virtual void load_state(Expr<uint> state_id) noexcept = 0;
        [[nodiscard]] virtual Float generate_1d() noexcept = 0;
        [[nodiscard]] virtual Float2 generate_2d() noexcept = 0;
        [[nodiscard]] virtual Float2 generate_pixel_2d() noexcept { return generate_2d(); }
    };

public:
    Sampler(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
    [[nodiscard]] auto seed() const noexcept { return _seed; }
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Sampler::Instance)

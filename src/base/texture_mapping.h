//
// Created by Mike Smith on 2022/3/11.
//

#pragma once

#include <dsl/syntax.h>
#include <util/command_buffer.h>
#include <base/scene_node.h>

namespace luisa::render {

using compute::Float;
using compute::Float2;

class Pipeline;
class Interaction;

class TextureMapping : public SceneNode {

public:
    struct Coord2D {
        Float2 st;
        Float ds_dx;
        Float ds_dy;
        Float dt_dx;
        Float dt_dy;
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const TextureMapping *_mapping;

    public:
        Instance(const Pipeline &pipeline, const TextureMapping *mapping) noexcept
            : _pipeline{pipeline}, _mapping{mapping} {}
        template<typename T = TextureMapping>
            requires std::is_base_of_v<TextureMapping, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_mapping); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Coord2D map(const Interaction &it, Expr<float> time) const noexcept = 0;
    };

public:
    TextureMapping(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::TextureMapping::Instance)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::TextureMapping::Coord2D)

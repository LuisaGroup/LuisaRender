//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <scene/scene_node.h>
#include <scene/sampler.h>

namespace luisa::render {

using compute::BindlessArray;

class Shape;
class Sampler;
class Interaction;

class Material : public SceneNode {

public:
    static constexpr auto property_flag_black = 1u;
    static constexpr auto property_flag_reflective = 2u;
    static constexpr auto property_flag_refractive = 4u;
    // TODO: more flags

public:
    struct Evaluation {
        Float3 f;
        Float pdf;
    };

    struct Sample {
        Float3 wi;
        Evaluation eval;
    };

    struct Closure {
        virtual ~Closure() noexcept = default;
        [[nodiscard]] virtual Evaluation evaluate(Expr<float3> wi) const noexcept = 0;
        [[nodiscard]] virtual Sample sample(Sampler::Instance &sampler) const noexcept = 0;
    };

public:
    Material(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual uint property_flags() const noexcept = 0;
    [[nodiscard]] virtual uint /* bindless buffer id */ encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint instance_id, const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Closure> decode(const Pipeline &pipeline, const Interaction &it) const noexcept = 0;
};

}// namespace luisa::render

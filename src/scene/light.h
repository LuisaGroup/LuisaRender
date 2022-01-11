//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <runtime/bindless_array.h>
#include <scene/scene_node.h>
#include <scene/sampler.h>
#include <scene/shape.h>

namespace luisa::render {

using compute::Ray;
using compute::BindlessArray;

class Shape;
class Interaction;

class Light : public SceneNode {

public:
    static constexpr auto property_flag_black = 1u;

public:
    struct Evaluation {
        Float3 Le;
        Float pdf;
    };

    struct Sample {
        Evaluation eval;
        Float3 p_light;
        Var<Ray> shadow_ray;
    };

public:
    Light(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual float power(const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual uint property_flags() const noexcept = 0;
    [[nodiscard]] virtual uint /* bindless buffer id */ encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint instance_id, const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual Evaluation evaluate(
        const Pipeline &pipeline, const Interaction &it, Expr<float3> p_from) const noexcept = 0;
    [[nodiscard]] virtual Sample sample(
        const Pipeline &pipeline, Sampler::Instance &sampler,
        Expr<uint> light_inst_id, const Interaction &it_from) const noexcept = 0;
};

}

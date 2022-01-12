//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <rtx/ray.h>
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
    struct Evaluation {
        Float3 Le;
        Float pdf;
    };

    struct Sample {
        Evaluation eval;
        Float3 p_light;
        Var<Ray> shadow_ray;
    };

    struct Closure {
        virtual ~Closure() noexcept = default;
        [[nodiscard]] virtual Evaluation evaluate(Expr<float3> p_from) const noexcept = 0;
        [[nodiscard]] virtual Sample sample(Sampler::Instance &sampler, Expr<uint> light_inst_id) const noexcept = 0;
    };

public:
    Light(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual float power(const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual uint /* bindless buffer id */ encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint instance_id, const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Closure> decode(const Pipeline &pipeline, const Interaction &it) const noexcept = 0;
};

}

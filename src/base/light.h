//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <rtx/ray.h>
#include <runtime/bindless_array.h>
#include <util/spectrum.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/shape.h>

namespace luisa::render {

using compute::BindlessArray;
using compute::Ray;

class Shape;
class Interaction;

class Light : public SceneNode {

public:
    struct Evaluation {
        Float4 L;
        Float pdf;
    };

    struct Sample {
        Evaluation eval;
        Var<Ray> shadow_ray;
    };

    struct Closure {
        virtual ~Closure() noexcept = default;
        [[nodiscard]] virtual Evaluation evaluate(
            const Interaction &it_light, Expr<float3> p_from) const noexcept = 0;
        [[nodiscard]] virtual Sample sample(
            Sampler::Instance &sampler, Expr<uint> light_inst_id,
            const Interaction &it_from) const noexcept = 0;
    };

public:
    Light(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual bool is_virtual() const noexcept = 0;
    [[nodiscard]] virtual uint /* bindless buffer id */ encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint instance_id, const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
};

}// namespace luisa::render

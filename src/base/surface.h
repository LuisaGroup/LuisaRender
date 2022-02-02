//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/spectrum.h>
#include <base/scene_node.h>
#include <base/sampler.h>

namespace luisa::render {

using compute::BindlessArray;

class Shape;
class Sampler;
class Interaction;

class Surface : public SceneNode {

public:
    struct Evaluation {
        SampledWavelengths swl;
        Float4 f;
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
    Surface(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] virtual uint /* bindless buffer id */ encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint instance_id, const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
};

}// namespace luisa::render

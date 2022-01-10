//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <runtime/bindless_array.h>
#include <scene/scene_node.h>
#include <scene/sampler.h>

namespace luisa::render {

using compute::BindlessArray;

class Shape;
class Interaction;

class Light : public SceneNode {

public:
    static constexpr auto property_flag_black = 1u;

public:
    struct Evaluation {
        Float3 Le;
        Float3 n;
        Float pdf;
    };

    struct Sample {
        Evaluation eval;
        Float3 p;
    };

public:
    Light(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual float power(const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual uint property_flags() const noexcept = 0;
    [[nodiscard]] virtual uint /* bindless buffer id */ encode(Pipeline &pipeline, CommandBuffer &command_buffer, const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual Evaluation evaluate(const Interaction &it) const noexcept = 0;
    [[nodiscard]] virtual Sample sample(Sampler::Instance &sampler, const Interaction &it) const noexcept = 0;
};

}

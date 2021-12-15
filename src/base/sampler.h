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
    Sampler(Scene *scene, const SceneNodeDesc *desc) noexcept;

    // start the sample sequence at a given pixel
    virtual void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept = 0;
    // save the sampler's state to its internal buffer
    virtual void save() noexcept = 0;
    // resume the sample sequence, from the previously saved state (in the internal buffer)
    virtual void resume(Expr<uint2> pixel) noexcept = 0;
    // generate a 1D sample, the internal (on-stack) state will be updated
    [[nodiscard]] virtual Float generate_1d() noexcept = 0;
    // generate a 2D sample, the internal (on-stack) state will be updated
    [[nodiscard]] virtual Float2 generate_2d() noexcept = 0;
};

}

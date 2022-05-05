//
// Created by ChenXin on 2022/4/11.
//

#pragma once

#include <luisa-compute.h>

#include <base/scene_node.h>

namespace luisa::render {

enum class Optimizer {
    BGD = 0,
    SGD = 1,
    AdaGrad = 2,
    Adam = 3,
    LDGD = 4,
};

class OptimizerTemp : public SceneNode {

public:
    OptimizerTemp(Scene *scene, const SceneNodeDesc *desc) noexcept;
    virtual void materialize(CommandBuffer &command_buffer) noexcept = 0;
    virtual void clear_gradients(CommandBuffer &command_buffer) noexcept = 0;
    virtual void apply_gradients(CommandBuffer &command_buffer, float alpha) noexcept = 0;
    /// Apply then clear the gradients
    void step(CommandBuffer &command_buffer, float alpha) noexcept {
        apply_gradients(command_buffer, alpha);
        clear_gradients(command_buffer);
    }
};

}// namespace luisa::render
//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <core/basic_types.h>
#include <base/scene_node.h>

namespace luisa::render {

class Transform : public SceneNode {

public:
    Transform(Scene *scene, const SceneDescNode *desc) noexcept
        : SceneNode{scene, desc, SceneNode::Tag::TRANSFORM} {}
    [[nodiscard]] virtual bool is_static() const noexcept = 0;
    [[nodiscard]] virtual float4x4 matrix(float time) const noexcept = 0;
};

}

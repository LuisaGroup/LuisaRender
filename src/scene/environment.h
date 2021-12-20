//
// Created by Mike Smith on 2021/12/15.
//

#pragma once

#include <scene/scene_node.h>

namespace luisa::render {

class Transform;

class Environment : public SceneNode {

private:
    const Transform *_transform;

public:
    Environment(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto transform() const noexcept { return _transform; }
};

}// namespace luisa::render

//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <base/scene_node.h>

namespace luisa::render {

class Material : public SceneNode {
public:
    Material(Scene *scene, const SceneNodeDesc *desc) noexcept;
};

}

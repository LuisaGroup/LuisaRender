//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <base/scene_node.h>

namespace luisa::render {

class Light : public SceneNode {
public:
    Light(Scene *scene, const SceneNodeDesc *desc) noexcept;
};

}

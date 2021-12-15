//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <base/scene_node.h>

namespace luisa::render {

class Filter : public SceneNode {

private:
    float2 _radius;

public:
    Filter(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto radius() const noexcept { return _radius; }
};

}

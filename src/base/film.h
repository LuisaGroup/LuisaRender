//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <core/basic_types.h>
#include <base/scene_node.h>

namespace luisa::render {

class Film : public SceneNode {

private:
    uint2 _resolution;

public:
    Film(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto resolution() const noexcept { return _resolution; }
};

}

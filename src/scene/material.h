//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <scene/scene_node.h>

namespace luisa::render {

class Material : public SceneNode {

public:
    struct Evaluation {
        Float3 f;
        Float pdf;
    };

    struct Sample {
        Float3 wi;
        Evaluation eval;
    };

public:
    Material(Scene *scene, const SceneNodeDesc *desc) noexcept;
};

}

//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <scene/scene_node.h>

namespace luisa::render {

class Shape;

struct LightInstance {

};

class Light : public SceneNode {

public:
    struct Evaluation {

    };

    struct Sample {

    };

    class Instance {

    };

public:
    Light(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual float power(const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Stream &stream, Pipeline &pipeline) const noexcept = 0;
};

}

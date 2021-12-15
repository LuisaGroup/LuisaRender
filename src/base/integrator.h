//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <base/scene_node.h>

namespace luisa::render {

class Sampler;

class Integrator : public SceneNode {

private:
    Sampler *_sampler;

public:
    Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
};

}

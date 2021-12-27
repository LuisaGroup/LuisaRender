//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <core/basic_types.h>
#include <scene/scene_node.h>

namespace luisa::render {

class Film : public SceneNode {

public:
    struct Instance : public SceneNode::Instance {
        virtual void accumulate(Expr<uint2> pixel, Expr<float3> color) const noexcept = 0;
    };

private:
    uint2 _resolution;

public:
    Film(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto resolution() const noexcept { return _resolution; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Stream &stream, Pipeline &pipeline) const noexcept = 0;
};

}

//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <base/scene_node.h>
#include <base/sampler.h>

namespace luisa::render {

class Filter : public SceneNode {

public:
    struct Sample {
        Float2 offset;
        Float weight;
        Float pdf;
    };

    struct Instance : public SceneNode::Instance {
        [[nodiscard]] virtual Sample sample_pixel(
            Sampler::Instance &sampler) const noexcept = 0;
    };

private:
    float2 _radius;

public:
    Filter(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto radius() const noexcept { return _radius; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Stream &stream) const noexcept = 0;
};

}

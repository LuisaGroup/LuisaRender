//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <scene/scene_node.h>
#include <scene/sampler.h>

namespace luisa::render {

class Filter : public SceneNode {

public:
    struct Sample {
        Float2 offset;
        Float weight;
        Float pdf;
    };

    class Instance : public SceneNode::Instance {

    private:
        const Filter *_filter;

    public:
        explicit Instance(const Filter *filter) noexcept : _filter{filter} {}
        [[nodiscard]] auto node() const noexcept { return _filter; }
        [[nodiscard]] virtual Sample sample_pixel(Sampler::Instance &sampler) const noexcept = 0;
    };

private:
    float2 _radius;

public:
    Filter(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto radius() const noexcept { return _radius; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}

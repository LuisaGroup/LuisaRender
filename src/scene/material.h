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

    class Instance {

    public:
        virtual ~Instance() noexcept = default;
        [[nodiscard]] virtual uint /* bindless buffer id */ encode_data(Stream &stream, Pipeline &pipeline) const noexcept = 0;
    };

public:
    Material(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Stream &stream, Pipeline &pipeline) const noexcept = 0;
};

}// namespace luisa::render

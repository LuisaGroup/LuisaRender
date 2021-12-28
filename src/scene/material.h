//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <scene/scene_node.h>

namespace luisa::render {

class Material : public SceneNode {

public:
    static constexpr auto property_flag_black = 1u;
    static constexpr auto property_flag_two_sided = 2u;
    static constexpr auto property_flag_reflective = 4u;
    static constexpr auto property_flag_refractive = 8u;
    static constexpr auto property_flag_volumetric = 16u;

public:
    struct Evaluation {
        Float3 f;
        Float pdf;
    };

    struct Sample {
        Float3 wi;
        Evaluation eval;
    };

    class Interface {

    public:
        virtual ~Interface() noexcept = default;
    };

public:
    Material(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Interface> interface() const noexcept = 0;
    [[nodiscard]] virtual uint property_flags() const noexcept = 0;
    [[nodiscard]] virtual uint /* bindless buffer id and tag */ encode(Stream &stream, Pipeline &pipeline) const noexcept = 0;
};

}// namespace luisa::render

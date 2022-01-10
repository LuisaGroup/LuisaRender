//
// Created by Mike Smith on 2022/1/10.
//

#pragma once

#include <scene/scene_node.h>

namespace luisa::render {

class LightDistribution : public SceneNode {

public:
    class Instance {

    private:
        const LightDistribution *_light_dist;

    public:
        explicit Instance(const LightDistribution *light_dist) noexcept : _light_dist{light_dist} {}
        [[nodiscard]] auto node() const noexcept { return _light_dist; }
        virtual ~Instance() noexcept = default;
    };

private:


public:
    LightDistribution(Scene *scene, const SceneNodeDesc *desc) noexcept;
};

}// namespace luisa::render

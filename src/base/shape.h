//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <base/scene_node.h>

namespace luisa::render {

class Light;
class Material;
class Transform;

class Shape : public SceneNode {

private:
    const Material *_material;
    const Light *_light;
    const Transform *_transform;

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto material() const noexcept { return _material; }
    [[nodiscard]] auto light() const noexcept { return _light; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] virtual bool is_static() const noexcept = 0;
};

}// namespace luisa::render

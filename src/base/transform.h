//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <core/basic_types.h>
#include <base/scene.h>

namespace luisa::render {

class Transform : public Scene::Node {

public:
    Transform() noexcept : Scene::Node{Scene::Node::Tag::TRANSFORM} {}
    [[nodiscard]] virtual bool is_static() const noexcept = 0;
    [[nodiscard]] virtual float4x4 matrix(float time) const noexcept = 0;
};

}

//
// Created by Mike on 2021/12/15.
//

#include <base/transform.h>

namespace luisa::render {

Transform::Transform(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TRANSFORM} {}

}// namespace luisa::render

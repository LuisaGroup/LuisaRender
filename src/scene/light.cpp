//
// Created by Mike on 2021/12/15.
//

#include <scene/light.h>

namespace luisa::render {

Light::Light(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LIGHT} {}

}// namespace luisa::render

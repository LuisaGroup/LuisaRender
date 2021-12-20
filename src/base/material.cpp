//
// Created by Mike on 2021/12/14.
//

#include <base/material.h>

namespace luisa::render {

Material::Material(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::MATERIAL} {}

}// namespace luisa::render

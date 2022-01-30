//
// Created by Mike on 2021/12/14.
//

#include <base/surface.h>

namespace luisa::render {

Surface::Surface(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SURFACE} {}

}// namespace luisa::render

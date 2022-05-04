//
// Created by ChenXin on 2022/5/4.
//

#include <base/loss.h>

namespace luisa::render {

Loss::Loss(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LOSS} {}

}// namespace luisa::render
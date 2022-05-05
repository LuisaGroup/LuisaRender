//
// Created by ChenXin on 2022/5/5.
//

#include <base/loss.h>

namespace luisa::render {

OptimizerTemp::OptimizerTemp(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::OPTIMIZER} {}

}// namespace luisa::render
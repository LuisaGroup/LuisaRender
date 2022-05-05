//
// Created by ChenXin on 2022/5/5.
//

#include <base/optimizer.h>

namespace luisa::render {

Optimizer::Optimizer(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::OPTIMIZER},
      _learning_rate{std::max(desc->property_float_or_default("learning_rate", 0.1f), 0.f)} {}

}// namespace luisa::render
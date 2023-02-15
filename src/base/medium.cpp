//
// Created by ChenXin on 2023/2/13.
//

#include "medium.h"

namespace luisa::render {

Medium::Medium(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::MEDIUM} {
    _priority = desc->property_uint_or_default("priority", 0u);
    _eta = desc->property_float_or_default("eta", 1.f);
}

unique_ptr<Medium::Instance> Medium::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto instance = _build(pipeline, command_buffer);
    return instance;
}

}
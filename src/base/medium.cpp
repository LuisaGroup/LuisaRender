//
// Created by ChenXin on 2023/2/13.
//

#include "medium.h"

namespace luisa::render {

Medium::Medium(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::MEDIUM}{}

unique_ptr<Medium::Instance> Medium::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return _build(pipeline, command_buffer);
}

}
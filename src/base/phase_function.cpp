//
// Created by ChenXin on 2023/2/14.
//

#include "phase_function.h"

namespace luisa::render {

PhaseFunction::PhaseFunction(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode(scene, desc, SceneNodeTag::PHASE_FUNCTION) {}

unique_ptr<PhaseFunction::Instance> PhaseFunction::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return _build(pipeline, command_buffer);
}

}// namespace luisa::render
//
// Created by Mike on 2021/12/8.
//

#include <base/sampler.h>

namespace luisa::render {

Sampler::Sampler(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SAMPLER} {}

}// namespace luisa::render

//
// Created by Mike on 2021/12/8.
//

#include <base/sampler.h>

namespace luisa::render {

Sampler::Sampler(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SAMPLER},
      _seed{desc->property_uint_or_default("seed", 19980810u)} {}

}// namespace luisa::render

//
// Created by Mike Smith on 2022/1/10.
//

#include <scene/light_distribution.h>

namespace luisa::render {

LightDistribution::LightDistribution(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LIGHT_DISTRIBUTION} {}

}

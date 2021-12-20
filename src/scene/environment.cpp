//
// Created by Mike Smith on 2021/12/15.
//

#include <scene/scene.h>
#include <scene/transform.h>
#include <sdl/scene_node_desc.h>
#include <scene/environment.h>

namespace luisa::render {

Environment::Environment(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::ENVIRONMENT},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))} {}

}// namespace luisa::render

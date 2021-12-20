//
// Created by Mike Smith on 2021/12/15.
//

#include <base/scene.h>
#include <base/transform.h>
#include <sdl/scene_node_desc.h>
#include <base/environment.h>

namespace luisa::render {

Environment::Environment(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::ENVIRONMENT},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))} {}

}// namespace luisa::render

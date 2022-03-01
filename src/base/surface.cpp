//
// Created by Mike on 2021/12/14.
//

#include <base/surface.h>
#include <base/scene.h>
#include <base/interaction.h>

namespace luisa::render {

Surface::Surface(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SURFACE},
      _normal_map{scene->load_texture(desc->property_node_or_default(
          "normal", SceneNodeDesc::shared_default(
                        SceneNodeTag::TEXTURE, "ConstGeneric")))} {}

}// namespace luisa::render

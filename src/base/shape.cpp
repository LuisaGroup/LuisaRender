//
// Created by Mike on 2021/12/14.
//

#include <base/material.h>
#include <base/light.h>
#include <base/transform.h>
#include <base/scene.h>
#include <base/scene_node_desc.h>
#include <base/shape.h>

namespace luisa::render {

Shape::Shape(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNode::Tag::SHAPE},
      _material{scene->load_material(desc->property_node_or_default("material"))},
      _light{scene->load_light(desc->property_node_or_default("light"))},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))} {}

}// namespace luisa::render

//
// Created by Mike on 2021/12/8.
//

#include <base/scene.h>
#include <base/film.h>
#include <base/filter.h>
#include <base/transform.h>
#include <base/scene_node_desc.h>
#include <base/camera.h>

namespace luisa::render {

Camera::Camera(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNode::Tag::CAMERA},
      _film{scene->load_film(desc->property_node_or_default("film"))},
      _filter{scene->load_filter(desc->property_node_or_default("filter"))},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))} {}

}// namespace luisa::render

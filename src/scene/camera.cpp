//
// Created by Mike on 2021/12/8.
//

#include <scene/scene.h>
#include <scene/film.h>
#include <scene/filter.h>
#include <scene/transform.h>
#include <sdl/scene_node_desc.h>
#include <scene/camera.h>

namespace luisa::render {

Camera::Camera(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::CAMERA},
      _film{scene->load_film(desc->property_node_or_default("film"))},
      _filter{scene->load_filter(desc->property_node_or_default("filter"))},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))} {}

}// namespace luisa::render

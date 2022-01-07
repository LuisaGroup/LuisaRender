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
      _film{scene->load_film(desc->property_node("film"))},
      _filter{scene->load_filter(desc->property_node_or_default("filter"))},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))},
      _time_span{desc->property_float2_or_default("time_span", luisa::make_float2())} {
    if (_time_span.y < _time_span.x) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid time span: [{}, {}].",
            _time_span.x, _time_span.y);
    }
}

}// namespace luisa::render

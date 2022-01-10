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

[[nodiscard]] static auto default_filter_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__default_filter__", SceneNodeTag::FILTER};
        d.define(SceneNodeTag::FILTER, "Box", {});
        return &d;
    }();
    return desc;
}

[[nodiscard]] static auto default_transform_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__default_transform__", SceneNodeTag::TRANSFORM};
        d.define(SceneNodeTag::TRANSFORM, "Identity", {});
        return &d;
    }();
    return desc;
}

Camera::Camera(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::CAMERA},
      _film{scene->load_film(desc->property_node("film"))},
      _filter{scene->load_filter(desc->property_node_or_default(
          "filter", default_filter_node_desc()))},
      _transform{scene->load_transform(desc->property_node_or_default(
          "transform", default_transform_node_desc()))},
      _time_span{desc->property_float2_or_default("time_span", luisa::make_float2())},
      _spp{desc->property_uint_or_default("spp", 1024u)},
      _file{desc->property_path_or_default(
          "file", std::filesystem::canonical(
                      desc->source_location() ?
                          desc->source_location().file()->parent_path() :
                          std::filesystem::current_path()) /
                      "color.exr")} {
    if (_time_span.y < _time_span.x) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid time span: [{}, {}].",
            _time_span.x, _time_span.y);
    }
}

}// namespace luisa::render

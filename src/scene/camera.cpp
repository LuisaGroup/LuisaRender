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
        static SceneNodeDesc d{"__camera_default_filter__", SceneNodeTag::FILTER};
        d.define(SceneNodeTag::FILTER, "Box", {});
        return &d;
    }();
    return desc;
}

[[nodiscard]] static auto default_transform_node_desc() noexcept {
    static auto desc = [] {
        static SceneNodeDesc d{"__camera_default_transform__", SceneNodeTag::TRANSFORM};
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
      _shutter_span{desc->property_float2_or_default(
          "shutter_span", [](auto desc) noexcept {
              return make_float2(desc->property_float_or_default(
                  "shutter_span", 0.0f));
          })},
      _shutter_samples{desc->property_uint_or_default("shutter_samples", 0u)},// 0 means default
      _spp{desc->property_uint_or_default("spp", 1024u)} {

    if (_shutter_span.y < _shutter_span.x) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid time span: [{}, {}].",
            _shutter_span.x, _shutter_span.y);
    }
    if (_shutter_samples == 0u) {
        _shutter_samples = std::max(_spp, 1024u);
    } else if (_shutter_samples > _spp) {
        _shutter_samples = _spp;
    }
    auto shutter_time_points = desc->property_float_list_or_default(
        "shutter_time_points", {_shutter_span.x, _shutter_span.y});
    auto shutter_weights = desc->property_float_list_or_default(
        "shutter_weights", {1.0f, 1.0f});
    if (shutter_time_points.size() != shutter_weights.size()) [[unlikely]] {
    }
    // TODO: process shutter curve

    // render file
    _file = desc->property_path_or_default(
        "file", std::filesystem::canonical(
                    desc->source_location() ?
                        desc->source_location().file()->parent_path() :
                        std::filesystem::current_path()) /
                    "color.exr");
    if (auto folder = _file.parent_path();
        !std::filesystem::exists(folder)) {
        std::filesystem::create_directories(folder);
    }
}

auto Camera::shutter_weight(float time) const noexcept -> float {
    if (time <= _shutter_span.x || time >= _shutter_span.y) { return 0.0f; }

    return 0.0f;
}

auto Camera::shutter_samples() const noexcept -> vector<ShutterSample> {
    return vector<ShutterSample>();
}

}// namespace luisa::render

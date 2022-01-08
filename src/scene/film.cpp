//
// Created by Mike on 2021/12/14.
//

#include <scene/film.h>

namespace luisa::render {

Film::Film(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::FILM},
      _resolution{desc->property_uint2_or_default(
          "resolution", make_uint2(desc->property_uint_or_default(
                            "resolution", 1024u)))},
      _spp{desc->property_uint_or_default("spp", 1024u)},
      _file{desc->property_path_or_default(
          "file", std::filesystem::canonical(
                      desc->source_location() ?
                          desc->source_location().file()->parent_path() :
                          std::filesystem::current_path()) /
                      "color.exr")} {}

}// namespace luisa::render

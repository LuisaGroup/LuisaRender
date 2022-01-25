//
// Created by Mike on 2021/12/14.
//

#include <base/film.h>

namespace luisa::render {

Film::Film(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::FILM},
      _resolution{desc->property_uint2_or_default(
          "resolution", lazy_construct([desc] {
              return make_uint2(desc->property_uint_or_default("resolution", 1024u));
          }))} {}

}// namespace luisa::render

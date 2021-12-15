//
// Created by Mike on 2021/12/14.
//

#include <base/film.h>

namespace luisa::render {

Film::Film(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNode::Tag::FILM} {}

}// namespace luisa::render

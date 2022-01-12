//
// Created by Mike on 2021/12/14.
//

#include <scene/material.h>
#include <scene/light.h>
#include <scene/transform.h>
#include <scene/scene.h>
#include <sdl/scene_node_desc.h>
#include <scene/shape.h>

namespace luisa::render {

Shape::Shape(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SHAPE},
      _material{scene == nullptr ? nullptr : scene->load_material(desc->property_node_or_default("material"))},
      _light{scene == nullptr ? nullptr : scene->load_light(desc->property_node_or_default("light"))},
      _transform{scene == nullptr ? nullptr : scene->load_transform(desc->property_node_or_default("transform"))} {
    if (desc->has_property("two_sided")) {
        _two_sided = desc->property_bool("two_sided");
    }
    auto hint = desc->property_string_or_default("build_hint", "");
    if (hint == "fast_update") {
        _build_hint = AccelBuildHint::FAST_UPDATE;
    } else if (hint == "fast_rebuild") {
        _build_hint = AccelBuildHint::FAST_REBUILD;
    }
}

}// namespace luisa::render

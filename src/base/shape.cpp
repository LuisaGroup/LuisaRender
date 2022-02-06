//
// Created by Mike on 2021/12/14.
//

#include <sdl/scene_node_desc.h>
#include <base/texture.h>
#include <base/surface.h>
#include <base/light.h>
#include <base/transform.h>
#include <base/scene.h>
#include <base/shape.h>

namespace luisa::render {

static auto shape_default_alpha_texture() noexcept {
    static auto desc = []{
        static SceneNodeDesc d{"__shape_default_alpha_texture", SceneNodeTag::TEXTURE};
        d.define(SceneNodeTag::TEXTURE, "ConstGeneric", {});
        d.add_property("v", 1.0);
        return &d;
    }();
    return desc;
}

Shape::Shape(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SHAPE},
      _surface{nullptr},
      _light{nullptr},
      _transform{nullptr} {

    if (auto light = desc->property_node_or_default("light")) { _light = scene->load_light(light); }
    if (auto material = desc->property_node_or_default("surface")) { _surface = scene->load_surface(material); }
    if (auto transform = desc->property_node_or_default("transform")) { _transform = scene->load_transform(transform);}

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

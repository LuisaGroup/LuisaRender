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

Shape::Shape(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SHAPE},
      _surface{scene->load_surface(desc->property_node_or_default("surface"))},
      _light{scene->load_light(desc->property_node_or_default("light"))},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))},
      _two_sided{}, _two_sided_specified{},
      _shadow_terminator{std::clamp(desc->property_float_or_default("shadow_terminator", 0.0f), 0.f, 1.f)} {
    if (desc->has_property("two_sided")) {
        _two_sided = desc->property_bool("two_sided");
        _two_sided_specified = true;
    }
}

}// namespace luisa::render

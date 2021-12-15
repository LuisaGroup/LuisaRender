//
// Created by Mike on 2021/12/8.
//

#include <core/logging.h>
#include <base/scene_node_desc.h>
#include <base/filter.h>

namespace luisa::render {

Filter::Filter(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNode::Tag::FILTER},
      _radius{desc->property_float2_or_default(
          "radius",
          make_float2(desc->property_float_or_default("radius", 1.0f)))} {
    if (any(_radius <= 0.0f)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid filter radius: ({}, {}).",
            _radius.x, _radius.y);
    }
}

}// namespace luisa::render

//
// Created by Mike on 2021/12/13.
//

#include <core/logging.h>
#include <base/scene_desc_node.h>
#include <base/scene_node.h>

namespace luisa::render {

SceneNode::SceneNode(Scene *scene, const SceneDescNode *desc, SceneNode::Tag tag) noexcept
    : _scene{scene}, _tag{tag} {
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description "
            "node '{}' (type = {}::{}).",
            desc->identifier(),
            tag_description(desc->tag()),
            desc->impl_type());
    }
    if (!desc->is_internal() && desc->tag() != tag) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid tag {} of scene description "
            "node '{}' (expected {}).",
            tag_description(desc->tag()),
            desc->identifier(),
            tag_description(tag));
    }
}

}

//
// Created by Mike on 2021/12/13.
//

#include <core/logging.h>
#include <base/scene_node.h>

namespace luisa::render {

SceneNode::SceneNode(const Scene *scene, const SceneNodeDesc *desc, SceneNodeTag tag) noexcept
    : _scene{reinterpret_cast<intptr_t>(scene)}, _tag{tag} {
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description "
            "node '{}' (type = {}::{}).",
            desc->identifier(),
            scene_node_tag_description(desc->tag()),
            desc->impl_type());
    }
    if (!desc->is_internal() && desc->tag() != tag) [[unlikely]] {
        LUISA_ERROR(
            "Invalid tag {} of scene description "
            "node '{}' (expected {}). [{}]",
            scene_node_tag_description(desc->tag()),
            desc->identifier(),
            scene_node_tag_description(tag),
            desc->source_location().string());
    }
}

}

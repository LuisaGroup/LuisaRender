//
// Created by Mike on 2021/12/13.
//

#include <base/scene_desc_node.h>

namespace luisa::render {

void SceneDescNode::add(std::string_view name, SceneDescNode::value_list value) noexcept {
    if (!_properties.emplace(name, std::move(value)).second) {
        LUISA_ERROR_WITH_LOCATION(
            "Redefinition of property '{}' in "
            "scene description node '{}'.",
            name, _identifier);
    }
}

SceneDescNode *SceneDescNode::add_internal(std::string_view name, std::string_view impl_type) noexcept {
    auto identifier = fmt::format(
        "{}{}${}$", internal_node_identifier_prefix,
        _identifier, name);
    auto unique_node = luisa::make_unique<SceneDescNode>(identifier, SceneNode::Tag::INTERNAL, impl_type);
    auto node = _internal_nodes.emplace_back(std::move(unique_node)).get();
    add(name, luisa::vector<node_type>{node});
    return node;
}

}// namespace luisa::render

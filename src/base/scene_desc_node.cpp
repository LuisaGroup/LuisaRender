//
// Created by Mike on 2021/12/13.
//

#include <base/scene_desc_node.h>

namespace luisa::render {

void SceneDescNode::add_property(std::string_view name, SceneDescNode::value_list value) noexcept {
    if (!_properties.emplace(name, std::move(value)).second) {
        LUISA_ERROR_WITH_LOCATION(
            "Redefinition of property '{}' in "
            "scene description node '{}'.",
            name, _identifier);
    }
}

SceneDescNode *SceneDescNode::define_internal(std::string_view name, std::string_view impl_type, SourceLocation location) noexcept {
    auto unique_node = luisa::make_unique<SceneDescNode>(
        std::string_view{}, SceneNode::Tag::INTERNAL);
    auto node = _internal_nodes.emplace_back(std::move(unique_node)).get();
    node->set_impl_type(impl_type);
    node->set_source_location(location);
    add_property(name, luisa::vector<node_type>{node});
    return node;
}

}// namespace luisa::render

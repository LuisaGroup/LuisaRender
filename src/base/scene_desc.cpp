//
// Created by Mike on 2021/12/13.
//

#include <base/scene_desc.h>

namespace luisa::render {

const SceneDescNode *SceneDesc::node(std::string_view identifier) const noexcept {
    auto iter = _global_nodes.find(identifier);
    return iter != _global_nodes.cend() ? nullptr : iter->get();
}

void SceneDesc::declare(std::string_view identifier, SceneNode::Tag tag) noexcept {
    if (tag == SceneNode::Tag::INTERNAL) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring forward declaration of "
            "internal node '{}'.",
            identifier);
    } else if (identifier == root_node_identifier ||
               tag == SceneNode::Tag::ROOT) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring forward declaration of root node");
    } else {
        auto [iter, first_decl] = _global_nodes.emplace(
            lazy_construct([identifier, tag] {
                return luisa::make_unique<SceneDescNode>(
                    identifier, tag, std::string_view{});
            }));
        if (auto node = iter->get();
            !first_decl && node->tag() != tag) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Forward-declaration of node '{}' has "
                "a different tag '{}' from '{}' "
                "in previous declarations.",
                identifier, SceneNode::tag_description(tag),
                SceneNode::tag_description(node->tag()));
        }
    }
}

SceneDescNode *SceneDesc::define(std::string_view identifier, SceneNode::Tag tag, std::string_view impl_type) noexcept {
    if (identifier == root_node_identifier ||
        tag == SceneNode::Tag::ROOT) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Defining root node as a normal "
            "global node is not allowed. "
            "Please use SceneDescNode::define_root().");
    }
    if (tag == SceneNode::Tag::INTERNAL) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Defining internal node as a "
            "global node is not allowed.");
    }
    auto [iter, first_decl] = _global_nodes.emplace(
        lazy_construct([identifier, tag] {
            return luisa::make_unique<SceneDescNode>(
                identifier, tag, std::string_view{});
        }));
    auto node = iter->get();
    if (!first_decl) {
        if (node->is_defined()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Redefinition of node '{}' in scene description.",
                node->identifier());
        }
        if (node->tag() != tag) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Definition of node '{}' has a different tag '{}' "
                "from '{}' in previous declarations.",
                identifier, SceneNode::tag_description(tag),
                SceneNode::tag_description(node->tag()));
        }
    }
    node->set_impl_type(impl_type);
    return node;
}

SceneDescNode *SceneDesc::define_root() noexcept {
    if (_root.is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Redefinition of root node "
            "in scene description.");
    }
    _root.set_impl_type(root_node_identifier);
    return &_root;
}

}// namespace luisa::render

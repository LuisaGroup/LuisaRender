//
// Created by Mike on 2021/12/13.
//

#include <sstream>
#include <base/scene_desc.h>

namespace luisa::render {

const SceneDescNode *SceneDesc::node(std::string_view identifier) const noexcept {
    if (auto iter = _global_nodes.find(identifier);
        iter != _global_nodes.cend()) {
        return iter->get();
    }
    LUISA_ERROR_WITH_LOCATION(
        "Global node '{}' not found "
        "in scene description.",
        identifier);
}

void SceneDesc::declare(std::string_view identifier, SceneNode::Tag tag) noexcept {
    if (tag == SceneNode::Tag::INTERNAL) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid forward declaration of "
            "internal node '{}'.",
            identifier);
    }
    if (identifier == root_node_identifier ||
        tag == SceneNode::Tag::ROOT) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid forward declaration of root node");
    }
    auto [iter, first_decl] = _global_nodes.emplace(
        lazy_construct([identifier, tag] {
            return luisa::make_unique<SceneDescNode>(
                identifier, tag);
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

SceneDescNode *SceneDesc::define(
    std::string_view identifier, SceneNode::Tag tag,
    std::string_view impl_type, SceneDescNode::SourceLocation location) noexcept {

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
                identifier, tag);
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
    node->set_source_location(location);
    return node;
}

SceneDescNode *SceneDesc::define_root(SceneDescNode::SourceLocation location) noexcept {
    if (_root.is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Redefinition of root node "
            "in scene description.");
    }
    _root.set_impl_type(root_node_identifier);
    _root.set_source_location(location);
    return &_root;
}

//namespace detail {
//
//static void validate(const SceneDescNode *node, size_t depth) noexcept {
//    if (depth > 32u) [[unlikely]] {
//        LUISA_ERROR_WITH_LOCATION(
//            "Scene description is too deep. "
//            "Recursions in definitions?");
//    }
//    if (!node->is_defined()) [[unlikely]] {
//        LUISA_ERROR_WITH_LOCATION(
//            "Node '{}' is referenced but not defined "
//            "in the scene description.",
//            node->identifier());
//    }
//    for (auto &&[prop, values] : node->properties()) {
//        if (auto nodes = std::get_if<SceneDescNode::node_list>(&values)) {
//            for (auto n : *nodes) { validate(n, depth + 1u); }
//        }
//    }
//}
//
//}// namespace detail
//
//void SceneDesc::validate() const noexcept {
//    if (!_source_path_stack.empty()) [[unlikely]] {
//        std::ostringstream oss;
//        for (auto p : _source_path_stack) { oss << "\n"
//                                                << *p; }
//        LUISA_ERROR_WITH_LOCATION(
//            "Unbalanced path stack in scene description. "
//            "Remaining paths (from stack top to bottom): {}",
//            oss.str());
//    }
//    detail::validate(&_root, 0u);
//}

void SceneDesc::push_source_path(const std::filesystem::path &path) noexcept {
    auto canonical_path = luisa::make_unique<std::filesystem::path>(
        std::filesystem::canonical(path));
    _source_path_stack.emplace_back(
        _source_paths.emplace_back(std::move(canonical_path)).get());
}

void SceneDesc::pop_source_path() noexcept {
    _source_path_stack.pop_back();
}

const std::filesystem::path *SceneDesc::current_source_path() const noexcept {
    if (_source_path_stack.empty()) { return nullptr; }
    return _source_path_stack.back();
}

}// namespace luisa::render

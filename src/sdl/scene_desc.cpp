//
// Created by Mike on 2021/12/13.
//

#include <sstream>
#include <sdl/scene_desc.h>

namespace luisa::render {

const SceneNodeDesc *SceneDesc::node(std::string_view identifier) const noexcept {
    if (auto iter = _global_nodes.find(identifier);
        iter != _global_nodes.cend()) {
        return iter->get();
    }
    LUISA_ERROR_WITH_LOCATION(
        "Global node '{}' not found "
        "in scene description.",
        identifier);
}

const SceneNodeDesc *SceneDesc::reference(std::string_view identifier) noexcept {
    if (identifier == root_node_identifier) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid reference to root node.");
    }
    return _global_nodes.emplace(
        lazy_construct([identifier] {
            return luisa::make_unique<SceneNodeDesc>(
                identifier, SceneNodeTag::DECLARATION);
        })).first->get();
}

SceneNodeDesc *SceneDesc::define(
    std::string_view identifier, SceneNodeTag tag,
    std::string_view impl_type, SceneNodeDesc::SourceLocation location) noexcept {

    if (identifier == root_node_identifier ||
        tag == SceneNodeTag::ROOT) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Defining root node as a normal "
            "global node is not allowed. "
            "Please use SceneNodeDesc::define_root().");
    }
    if (tag == SceneNodeTag::INTERNAL ||
        tag == SceneNodeTag::DECLARATION) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Defining internal or declaration node "
            "as a global node is not allowed.");
    }
    auto [iter, first_decl] = _global_nodes.emplace(
        lazy_construct([identifier, tag] {
            return luisa::make_unique<SceneNodeDesc>(
                identifier, tag);
        }));
    auto node = iter->get();
    if (!first_decl) {
        if (node->is_defined()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Redefinition of node '{}' in scene description.",
                node->identifier());
        }
        if (node->tag() != SceneNodeTag::DECLARATION &&
            node->tag() != tag) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Definition of node '{}' has a different tag '{}' "
                "from '{}' in previous declarations.",
                identifier, scene_node_tag_description(tag),
                scene_node_tag_description(node->tag()));
        }
    }
    node->set_tag(tag);
    node->set_impl_type(impl_type);
    node->set_source_location(location);
    return node;
}

SceneNodeDesc *SceneDesc::define_root(SceneNodeDesc::SourceLocation location) noexcept {
    if (_root.is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Redefinition of root node "
            "in scene description.");
    }
    _root.set_impl_type(root_node_identifier);
    _root.set_source_location(location);
    return &_root;
}

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

//
// Created by Mike on 2021/12/13.
//

#include <mutex>
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
    std::scoped_lock lock{_mutex};
    auto [iter, _] = _global_nodes.emplace(
        lazy_construct([identifier] {
            return luisa::make_unique<SceneNodeDesc>(
                identifier, SceneNodeTag::DECLARATION);
        }));
    return iter->get();
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

    std::scoped_lock lock{_mutex};
    auto [iter, _] = _global_nodes.emplace(
        lazy_construct([identifier, tag] {
            return luisa::make_unique<SceneNodeDesc>(
                identifier, tag);
        }));
    auto node = iter->get();
    if (node->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Redefinition of node '{}' ({}::{}) "
            "in scene description.",
            node->identifier(),
            scene_node_tag_description(node->tag()),
            node->impl_type());
    }
    node->define(tag, impl_type, location);
    return node;
}

SceneNodeDesc *SceneDesc::define_root(SceneNodeDesc::SourceLocation location) noexcept {
    std::scoped_lock lock{_mutex};
    if (_root.is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Redefinition of root node "
            "in scene description.");
    }
    _root.define(SceneNodeTag::ROOT, root_node_identifier, location);
    return &_root;
}

const std::filesystem::path *SceneDesc::register_path(std::filesystem::path path) noexcept {
    auto p = luisa::make_unique<std::filesystem::path>(std::move(path));
    std::scoped_lock lock{_mutex};
    return _paths.emplace_back(std::move(p)).get();
}

}// namespace luisa::render

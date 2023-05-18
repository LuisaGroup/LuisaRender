//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <mutex>
#include <sdl/scene_node_desc.h>

namespace luisa::render {

class SceneDesc {

private:
    [[nodiscard]] static auto _node_identifier(const luisa::unique_ptr<SceneNodeDesc> &node) noexcept { return node->identifier(); }
    [[nodiscard]] static auto _node_identifier(const SceneNodeDesc *node) noexcept { return node->identifier(); }
    [[nodiscard]] static auto _node_identifier(luisa::string_view s) noexcept { return s; }

public:
    struct NodeHash {
        using is_transparent = void;
        template<typename T>
        [[nodiscard]] auto operator()(T &&node) const noexcept -> uint64_t {
            return hash_value(_node_identifier(std::forward<T>(node)));
        }
    };
    struct NodeEqual {
        using is_transparent = void;
        template<typename Lhs, typename Rhs>
        [[nodiscard]] auto operator()(Lhs &&lhs, Rhs &&rhs) const noexcept -> bool {
            return _node_identifier(std::forward<Lhs>(lhs)) == _node_identifier(std::forward<Rhs>(rhs));
        }
    };
    static constexpr luisa::string_view root_node_identifier = "render";

private:
    luisa::unordered_set<luisa::unique_ptr<SceneNodeDesc>, NodeHash, NodeEqual> _global_nodes;
    luisa::vector<luisa::unique_ptr<std::filesystem::path>> _paths;
    SceneNodeDesc _root;
    std::recursive_mutex _mutex;

public:
    SceneDesc() noexcept : _root{luisa::string{root_node_identifier}, SceneNodeTag::ROOT} {}
    [[nodiscard]] auto &nodes() const noexcept { return _global_nodes; }
    [[nodiscard]] const SceneNodeDesc *node(luisa::string_view identifier) const noexcept;
    [[nodiscard]] auto root() const noexcept { return &_root; }
    [[nodiscard]] const SceneNodeDesc *reference(luisa::string_view identifier) noexcept;
    [[nodiscard]] SceneNodeDesc *define(
        luisa::string_view identifier, SceneNodeTag tag, luisa::string_view impl_type,
        SceneNodeDesc::SourceLocation location = {}, const SceneNodeDesc *base = nullptr) noexcept;
    [[nodiscard]] SceneNodeDesc *define_root(SceneNodeDesc::SourceLocation location = {}) noexcept;
    const std::filesystem::path *register_path(std::filesystem::path path) noexcept;
};

}// namespace luisa::render

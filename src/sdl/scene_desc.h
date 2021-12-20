//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <sdl/scene_node_desc.h>

namespace luisa::render {

class SceneDesc {

private:
    [[nodiscard]] static auto _node_identifier(const luisa::unique_ptr<SceneNodeDesc> &node) noexcept { return node->identifier(); }
    [[nodiscard]] static auto _node_identifier(const SceneNodeDesc *node) noexcept { return node->identifier(); }
    [[nodiscard]] static auto _node_identifier(std::string_view s) noexcept { return s; }

public:
    struct NodeHash {
        using is_transparent = void;
        template<typename T>
        [[nodiscard]] auto operator()(T &&node) const noexcept -> uint64_t {
            return hash64(_node_identifier(std::forward<T>(node)));
        }
    };
    struct NodeEqual {
        using is_transparent = void;
        template<typename Lhs, typename Rhs>
        [[nodiscard]] auto operator()(Lhs &&lhs, Rhs &&rhs) const noexcept {
            return _node_identifier(std::forward<Lhs>(lhs)) == _node_identifier(std::forward<Rhs>(rhs));
        }
    };
    static constexpr std::string_view root_node_identifier = "render";

private:
    luisa::unordered_set<luisa::unique_ptr<SceneNodeDesc>, NodeHash, NodeEqual> _global_nodes;
    luisa::vector<luisa::unique_ptr<std::filesystem::path>> _source_paths;
    luisa::vector<const std::filesystem::path *> _source_path_stack;
    SceneNodeDesc _root;

public:
    SceneDesc() noexcept: _root{root_node_identifier, SceneNodeTag::ROOT} {}
    [[nodiscard]] auto &nodes() const noexcept { return _global_nodes; }
    [[nodiscard]] const SceneNodeDesc *node(std::string_view identifier) const noexcept;
    [[nodiscard]] auto root() const noexcept { return &_root; }
    [[nodiscard]] const SceneNodeDesc *reference(std::string_view identifier) noexcept;
    [[nodiscard]] SceneNodeDesc *define(
        std::string_view identifier, SceneNodeTag tag,
        std::string_view impl_type, SceneNodeDesc::SourceLocation location = {}) noexcept;
    [[nodiscard]] SceneNodeDesc *define_root(SceneNodeDesc::SourceLocation location = {}) noexcept;
    void push_source_path(const std::filesystem::path &path) noexcept;
    void pop_source_path() noexcept;
    [[nodiscard]] const std::filesystem::path *current_source_path() const noexcept;
    template<typename F>
    void with_source_path(const std::filesystem::path &path, F &&f) noexcept {
        push_source_path(path);
        std::invoke(std::forward<F>(f));
        pop_source_path();
    }
};

}// namespace luisa::render

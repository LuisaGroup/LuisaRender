//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <base/scene_desc_node.h>

namespace luisa::render {

class SceneDesc {

private:
    [[nodiscard]] static auto _node_identifier(const luisa::unique_ptr<SceneDescNode> &node) noexcept { return node->identifier(); }
    [[nodiscard]] static auto _node_identifier(const SceneDescNode *node) noexcept { return node->identifier(); }
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
    luisa::unordered_set<luisa::unique_ptr<SceneDescNode>, NodeHash, NodeEqual> _global_nodes;
    std::filesystem::path _base_folder;
    SceneDescNode _root;

public:
    explicit SceneDesc(std::filesystem::path base_folder) noexcept
        : _base_folder{std::move(base_folder)},
          _root{root_node_identifier, SceneNode::Tag::ROOT, {}} {}
    [[nodiscard]] auto &base_directory() const noexcept { return _base_folder; }
    [[nodiscard]] auto &nodes() const noexcept { return _global_nodes; }
    [[nodiscard]] const SceneDescNode *node(std::string_view identifier) const noexcept;
    [[nodiscard]] auto root() const noexcept { return &_root; }
    void declare(std::string_view identifier, SceneNode::Tag tag) noexcept;
    [[nodiscard]] SceneDescNode *define(std::string_view identifier, SceneNode::Tag tag, std::string_view impl_type) noexcept;
    [[nodiscard]] SceneDescNode *define_root() noexcept;
    void validate() const noexcept;
};

}// namespace luisa::render

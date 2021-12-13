//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <variant>
#include <filesystem>

#include <core/hash.h>
#include <core/allocator.h>
#include <base/scene_node.h>

namespace luisa::render {

class SceneDesc;

class SceneDescNode {

public:
    using bool_type = bool;
    using number_type = double;
    using path_type = std::filesystem::path;
    using node_type = const SceneDescNode *;
    using value_list = std::variant<
        luisa::vector<bool_type>,
        luisa::vector<number_type>,
        luisa::vector<path_type>,
        luisa::vector<node_type>>;
    static constexpr std::string_view internal_node_identifier_prefix = "$internal$";

private:
    luisa::string _identifier;
    SceneNode::Tag _tag;
    luisa::string _impl_type;
    luisa::vector<luisa::unique_ptr<SceneDescNode>> _internal_nodes;
    luisa::unordered_map<luisa::string, value_list, Hash64> _properties;

public:
    SceneDescNode(std::string_view identifier, SceneNode::Tag tag, std::string_view impl_type) noexcept
        : _identifier{identifier}, _tag{tag}, _impl_type{impl_type} {}
    SceneDescNode(SceneDescNode &&) noexcept = delete;
    SceneDescNode(const SceneDescNode &) noexcept = delete;
    SceneDescNode &operator=(SceneDescNode &&) noexcept = delete;
    SceneDescNode &operator=(const SceneDescNode &) noexcept = delete;
    [[nodiscard]] auto identifier() const noexcept { return std::string_view{_identifier}; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto impl_type() const noexcept { return std::string_view{_impl_type}; }
    void set_impl_type(std::string_view t) noexcept { _impl_type = t; }
    [[nodiscard]] auto &properties() const noexcept { return _properties; }
    void add(std::string_view name, value_list value) noexcept;
    SceneDescNode *add_internal(std::string_view name, std::string_view impl_type) noexcept;
    [[nodiscard]] auto is_root() const noexcept { return _tag == SceneNode::Tag::ROOT; }
    [[nodiscard]] auto is_internal() const noexcept { return _tag == SceneNode::Tag::INTERNAL; }
    [[nodiscard]] auto is_defined() const noexcept { return !_impl_type.empty(); }
};

}// namespace luisa::render

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
    using string_type = luisa::string;
    using node_type = const SceneDescNode *;

    using bool_list = luisa::vector<bool_type>;
    using number_list = luisa::vector<number_type>;
    using string_list = luisa::vector<string_type>;
    using node_list = luisa::vector<node_type>;

    using value_list = std::variant<
        bool_list, number_list, string_list, node_list>;

    class SourceLocation {

    private:
        const std::filesystem::path *_file;
        uint32_t _line;
        uint32_t _column;

    public:
        SourceLocation() noexcept : _file{nullptr}, _line{}, _column{} {}
        SourceLocation(const std::filesystem::path *path, uint32_t line, uint32_t col) noexcept
            : _file{path}, _line{line}, _column{col} {}
        [[nodiscard]] explicit operator bool() const noexcept { return _file != nullptr; }
        [[nodiscard]] auto file() const noexcept { return _file; }
        [[nodiscard]] auto line() const noexcept { return _line; }
        [[nodiscard]] auto column() const noexcept { return _column; }
    };

private:
    luisa::string _identifier;
    SourceLocation _location;
    SceneNode::Tag _tag;
    luisa::string _impl_type;
    luisa::vector<luisa::unique_ptr<SceneDescNode>> _internal_nodes;
    luisa::unordered_map<luisa::string, value_list, Hash64> _properties;

public:
    SceneDescNode(std::string_view identifier, SceneNode::Tag tag) noexcept
        : _identifier{identifier}, _tag{tag} {}
    SceneDescNode(SceneDescNode &&) noexcept = delete;
    SceneDescNode(const SceneDescNode &) noexcept = delete;
    SceneDescNode &operator=(SceneDescNode &&) noexcept = delete;
    SceneDescNode &operator=(const SceneDescNode &) noexcept = delete;
    [[nodiscard]] auto identifier() const noexcept { return std::string_view{_identifier}; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto impl_type() const noexcept { return std::string_view{_impl_type}; }
    [[nodiscard]] auto source_location() const noexcept { return _location; }
    void set_impl_type(std::string_view t) noexcept { _impl_type = t; }
    void set_source_location(SourceLocation l) noexcept { _location = l; }
    [[nodiscard]] auto &properties() const noexcept { return _properties; }
    void add_property(std::string_view name, value_list values) noexcept;
    void add_property(std::string_view name, bool_type value) noexcept { add_property(name, bool_list{value}); }
    void add_property(std::string_view name, number_type value) noexcept { add_property(name, number_list{value}); }
    void add_property(std::string_view name, string_type value) noexcept { add_property(name, string_list{std::move(value)}); }
    void add_property(std::string_view name, node_type value) noexcept { add_property(name, node_list{value}); }
    [[nodiscard]] SceneDescNode *define_internal(std::string_view name, std::string_view impl_type, SourceLocation location = {}) noexcept;
    [[nodiscard]] auto is_root() const noexcept { return _tag == SceneNode::Tag::ROOT; }
    [[nodiscard]] auto is_internal() const noexcept { return _tag == SceneNode::Tag::INTERNAL; }
    [[nodiscard]] auto is_defined() const noexcept { return !_impl_type.empty(); }
};

}// namespace luisa::render

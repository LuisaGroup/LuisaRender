//
// Created by Mike Smith on 2021/12/21.
//

#include <string_view>
#include <filesystem>
#include <future>

#include <core/allocator.h>
#include <sdl/scene_node_desc.h>

namespace luisa::render {

class SceneDesc;
class SceneNodeDesc;

class SceneParser {

private:
    SceneDesc &_desc;
    SceneNodeDesc::SourceLocation _location;
    luisa::vector<std::future<void>> _import_parsing;
    luisa::string _source;
    size_t _cursor;

private:
    [[noreturn]] void _report_error(std::string_view message) const noexcept;
    void _report_warning(std::string_view message) const noexcept;
    void _match(char c) noexcept;
    void _skip() noexcept;
    [[nodiscard]] char _peek() noexcept;
    [[nodiscard]] char _get() noexcept;
    [[nodiscard]] bool _eof() const noexcept;
    [[nodiscard]] std::string_view _read_identifier() noexcept;
    [[nodiscard]] double _read_number() noexcept;
    [[nodiscard]] bool _read_bool() noexcept;
    [[nodiscard]] luisa::string _read_string() noexcept;
    void _skip_blanks() noexcept;
    void _parse() noexcept;
    void _parse_file() noexcept;
    void _parse_root_node(SceneNodeDesc::SourceLocation l) noexcept;
    void _parse_global_node(SceneNodeDesc::SourceLocation l, std::string_view tag_desc) noexcept;
    void _parse_node_body(SceneNodeDesc *node) noexcept;
    [[nodiscard]] SceneNodeDesc::value_list _parse_value_list(SceneNodeDesc *node) noexcept;
    [[nodiscard]] SceneNodeDesc::number_list _parse_number_list_values() noexcept;
    [[nodiscard]] SceneNodeDesc::bool_list _parse_bool_list_values() noexcept;
    [[nodiscard]] SceneNodeDesc::node_list _parse_node_list_values(SceneNodeDesc *node) noexcept;
    [[nodiscard]] SceneNodeDesc::string_list _parse_string_list_values() noexcept;

    SceneParser(SceneDesc &desc, const std::filesystem::path &path) noexcept;

public:
    SceneParser(SceneParser &&) noexcept = default;
    SceneParser(const SceneParser &) noexcept = delete;
    SceneParser &operator=(SceneParser &&) noexcept = delete;
    SceneParser &operator=(const SceneParser &) noexcept = delete;
    [[nodiscard]] static luisa::unique_ptr<SceneDesc> parse(const std::filesystem::path &entry_file) noexcept;
};

}// namespace luisa::render

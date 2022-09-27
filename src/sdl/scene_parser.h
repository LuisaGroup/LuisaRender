//
// Created by Mike Smith on 2021/12/21.
//

#include <string_view>
#include <filesystem>
#include <future>

#include <core/stl.h>
#include <sdl/scene_node_desc.h>

namespace luisa::render {

class SceneDesc;
class SceneNodeDesc;
class SceneParserJSON;

class SceneParser {

public:
    using MacroMap = luisa::map<luisa::string, luisa::string>;

private:
    SceneDesc &_desc;
    const MacroMap &_cli_macros;
    MacroMap _local_macros;
    luisa::vector<luisa::string_view> _parsing_macros;
    SceneNodeDesc::SourceLocation _location;
    luisa::string _source;
    size_t _cursor;

private:
    template<typename... Args>
    [[noreturn]] void _report_error(std::string_view format, Args &&...args) const noexcept;
    template<typename... Args>
    void _report_warning(std::string_view format, Args &&...args) const noexcept;

private:
    void _match(char c) noexcept;
    void _skip() noexcept;
    void _skip_blanks() noexcept;
    [[nodiscard]] char _peek(bool escape_macro = false) noexcept;
    [[nodiscard]] char _get(bool escape_macro = false) noexcept;
    [[nodiscard]] bool _eof() const noexcept;
    [[nodiscard]] luisa::string _read_identifier(bool escape_macro = false) noexcept;
    [[nodiscard]] double _read_number() noexcept;
    [[nodiscard]] bool _read_bool() noexcept;
    [[nodiscard]] luisa::string _read_string() noexcept;
    void _parse_macro() noexcept;
    void _parse_define() noexcept;
    void _parse_file() noexcept;
    void _parse_source() noexcept;
    void _parse_root_node(SceneNodeDesc::SourceLocation l) noexcept;
    void _parse_global_node(SceneNodeDesc::SourceLocation l, std::string_view tag_desc) noexcept;
    void _parse_node_body(SceneNodeDesc *node) noexcept;
    [[nodiscard]] SceneNodeDesc::value_list _parse_value_list(SceneNodeDesc *node) noexcept;
    [[nodiscard]] SceneNodeDesc::number_list _parse_number_list_values() noexcept;
    [[nodiscard]] SceneNodeDesc::bool_list _parse_bool_list_values() noexcept;
    [[nodiscard]] SceneNodeDesc::node_list _parse_node_list_values(SceneNodeDesc *node) noexcept;
    [[nodiscard]] SceneNodeDesc::string_list _parse_string_list_values() noexcept;
    [[nodiscard]] const SceneNodeDesc *_parse_base_node() noexcept;

    friend class SceneParserJSON;
    SceneParser(SceneDesc &desc, const std::filesystem::path &path,
                const MacroMap &cli_macros) noexcept;
    static void _dispatch_parse(SceneDesc &desc, const std::filesystem::path &path,
                                const MacroMap &cli_macros) noexcept;

public:
    SceneParser(SceneParser &&) noexcept = default;
    SceneParser(const SceneParser &) noexcept = delete;
    SceneParser &operator=(SceneParser &&) noexcept = delete;
    SceneParser &operator=(const SceneParser &) noexcept = delete;
    [[nodiscard]] static luisa::unique_ptr<SceneDesc> parse(
        const std::filesystem::path &entry_file,
        const MacroMap &cli_macros) noexcept;
};

}// namespace luisa::render

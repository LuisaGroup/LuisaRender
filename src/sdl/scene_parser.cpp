//
// Created by Mike Smith on 2021/12/21.
//

#include <fstream>
#include <streambuf>
#include <fast_float/fast_float.h>

#include <core/logging.h>
#include <util/thread_pool.h>
#include <sdl/scene_parser.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_parser_json.h>

namespace luisa::render {

inline SceneParser::SceneParser(SceneDesc &desc, const std::filesystem::path &path,
                                const MacroMap &cli_macros) noexcept
    : _desc{desc}, _cli_macros{cli_macros},
      _location{desc.register_path(LUISA_SCENE_PARSER_CHECKED_CANONICAL_PATH(path))},
      _cursor{0u} {}

void SceneParser::_dispatch_parse(SceneDesc &desc, const std::filesystem::path &path,
                                  const MacroMap &cli_macros) noexcept {
    auto ext = path.extension().string();
    for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
    if (ext == ".json") {
        SceneParserJSON p{desc, path, cli_macros};
        p.parse();
    } else {
        SceneParser p{desc, path, cli_macros};
        p._parse_file();
    }
}

template<typename... Args>
inline void SceneParser::_report_error(std::string_view format, Args &&...args) const noexcept {
    LUISA_ERROR("{} [{}]", fmt::format(format, std::forward<Args>(args)...), _location.string());
}

template<typename... Args>
inline void SceneParser::_report_warning(std::string_view format, Args &&...args) const noexcept {
    LUISA_WARNING("{} [{}]", fmt::format(format, std::forward<Args>(args)...), _location.string());
}

inline void SceneParser::_parse_file() noexcept {
    //    auto file_path = _location.file()->string();
    //    auto file = fopen(file_path.c_str(), "r");
    //    if (file == nullptr) {
    //        LUISA_ERROR_WITH_LOCATION(
    //            "Failed to open file '{}'.",
    //            file_path);
    //    }
    //    fseek(file, 0, SEEK_END);
    //    auto length = ftell(file);
    //    fseek(file, 0, SEEK_SET);
    //    _source.resize(length);
    //    fread(_source.data(), 1, length, file);
    //    fclose(file);
    //    _parse_source();
    //    _source.clear();
    //    _source.shrink_to_fit();
    std::ifstream file{*_location.file()};
    _source = {
        std::istreambuf_iterator<char>{file},
        std::istreambuf_iterator<char>{}};
    _parse_source();
    _source.clear();
    _source.shrink_to_fit();
}

inline void SceneParser::_parse_source() noexcept {
    _skip_blanks();
    while (!_eof()) {
        auto loc = _location;
        if (auto token = _read_identifier(); token == "import") {// import
            _skip_blanks();
            std::filesystem::path path{_read_string()};
            if (!path.is_absolute()) { path = _location.file()->parent_path() / path; }
            global_thread_pool().async([path = std::move(path), &desc = _desc, &cli_macros = _cli_macros] {
                SceneParser::_dispatch_parse(desc, path, cli_macros);
            });
        } else if (token == "define") {
            _parse_define();
        } else if (token == SceneDesc::root_node_identifier) {// root node
            _parse_root_node(loc);
        } else [[likely]] {// scene node
            _parse_global_node(loc, token);
        }
        _skip_blanks();
    }
}

inline void SceneParser::_match(char c) noexcept {
    if (auto got = _get(); got != c) [[unlikely]] {
        _report_error(
            "Invalid character '{}' "
            "(expected '{}').",
            got, c);
    }
}

void SceneParser::_skip() noexcept { static_cast<void>(_get(true)); }

inline char SceneParser::_peek(bool escape_macro) noexcept {
    auto peek_char = [this] {
        if (!_parsing_macros.empty()) {
            return _parsing_macros.back().front();
        }
        if (_eof()) [[unlikely]] { _report_error("Premature EOF."); }
        auto c = _source[_cursor];
        if (c == '\r') {
            if (_source[_cursor + 1u] == '\n') { _cursor++; }
            return '\n';
        }
        return c;
    };
    auto c = peek_char();
    if (!escape_macro) {
        while (c == '#') {
            _skip();
            _parse_macro();
            c = peek_char();
        }
    }
    return c;
}

inline char SceneParser::_get(bool escape_macro) noexcept {
    auto get_char = [this] {
        if (!_parsing_macros.empty()) {
            auto m = _parsing_macros.back();
            _parsing_macros.pop_back();
            auto c = m.front();
            if (m.size() > 1u) {
                _parsing_macros.emplace_back(m.substr(1u));
            }
            return c;
        }
        if (_eof()) [[unlikely]] { _report_error("Premature EOF."); }
        auto c = _source[_cursor++];
        if (c == '\r') {
            if (_source[_cursor] == '\n') { _cursor++; }
            _location.set_line(_location.line() + 1u);
            _location.set_column(0u);
            return '\n';
        }
        if (c == '\n') {
            _location.set_line(_location.line() + 1u);
            _location.set_column(0u);
        } else {
            _location.set_column(_location.column() + 1u);
        }
        return c;
    };
    auto c = get_char();
    if (!escape_macro) {
        while (c == '#') {
            _parse_macro();
            c = get_char();
        }
    }
    return c;
}

inline bool SceneParser::_eof() const noexcept {
    return _parsing_macros.empty() && _cursor >= _source.size();
}

inline luisa::string SceneParser::_read_identifier(bool escape_macro) noexcept {
    luisa::string identifier;
    auto c = _get(escape_macro);
    if (c != '$' && c != '_' && !isalpha(c)) [[unlikely]] {
        _report_error("Invalid character '{}' in identifier.", c);
    }
    identifier.push_back(c);
    auto is_ident_body = [](auto c) noexcept {
        return isalnum(c) || c == '_' || c == '$' || c == '-';
    };
    while (!_eof() && is_ident_body(_peek(escape_macro))) {
        identifier.push_back(_get(escape_macro));
    }
    return identifier;
}

inline double SceneParser::_read_number() noexcept {
    static thread_local luisa::string s;
    s.clear();
    if (auto c = _peek(); c == '+') [[unlikely]] {
        _skip();
        _skip_blanks();
    } else if (c == '-') {
        s.push_back(_get());
        _skip_blanks();
    }
    auto is_digit = [](auto c) noexcept { return isdigit(c) || c == '.' || c == 'e' || c == '-' || c == '+'; };
    while (!_eof() && is_digit(_peek())) { s.push_back(_get()); }
    auto value = 0.0;
    if (auto result = fast_float::from_chars(s.data(), s.data() + s.size(), value);
        result.ec != std::errc{}) [[unlikely]] {
        _report_error("Invalid number string '{}...'.", s.substr(0, 4));
    }
    return value;
}

inline bool SceneParser::_read_bool() noexcept {
    using namespace std::string_view_literals;
    if (_peek() == 't') {
        for (auto x : "true"sv) { _match(x); }
        return true;
    }
    for (auto x : "false"sv) { _match(x); }
    return false;
}

inline luisa::string SceneParser::_read_string() noexcept {
    auto quote = _get();
    if (quote != '"' && quote != '\'') [[unlikely]] {
        _report_error("Expected string but got {}.", quote);
    }
    luisa::string s;
    for (auto c = _get(); c != quote; c = _get()) {
        if (!isprint(c)) [[unlikely]] {
            _report_error(
                "Unexpected non-printable character 0x{:02x}.",
                static_cast<int>(c));
        }
        if (c == '\\') [[unlikely]] {// escape
            c = [this, esc = _get(true)] {
                switch (esc) {
                    case 'b': return '\b';
                    case 'f': return '\f';
                    case 'n': return '\n';
                    case 'r': return '\r';
                    case 't': return '\t';
                    case '\\': return '\\';
                    case '\'': return '\'';
                    case '"': return '\"';
                    case '#': return '#';
                    default: _report_error(
                        "Invalid escaped character '{}'.", esc);
                }
            }();
        }
        s.push_back(c);
    }
    return s;
}

inline void SceneParser::_skip_blanks() noexcept {
    while (!_eof()) {
        if (auto c = _peek(true); isblank(c) || c == '\n') {// blank
            _skip();
        } else if (c == '/') {// comment
            _skip();
            _match('/');
            while (!_eof() && _get(true) != '\n') {}
        } else {
            break;
        }
    }
}

inline void SceneParser::_parse_root_node(SceneNodeDesc::SourceLocation l) noexcept {
    _parse_node_body(_desc.define_root(l));
}

inline void SceneParser::_parse_global_node(SceneNodeDesc::SourceLocation l, std::string_view tag_desc) noexcept {
    auto tag = parse_scene_node_tag(tag_desc);
    if (tag == SceneNodeTag::ROOT) [[unlikely]] {
        _report_error(
            "Invalid scene node type '{}'.",
            tag_desc);
    }
    _skip_blanks();
    auto name = _read_identifier();
    _skip_blanks();
    const SceneNodeDesc *base = nullptr;
    luisa::string impl_type;
    if (_peek() == ':') {
        _match(':');
        _skip_blanks();
        impl_type = _read_identifier();
        _skip_blanks();
        if (_peek() == '(') { base = _parse_base_node(); }
        _skip_blanks();
    }
    _parse_node_body(_desc.define(name, tag, impl_type, l, base));
}

void SceneParser::_parse_node_body(SceneNodeDesc *node) noexcept {
    _skip_blanks();
    _match('{');
    _skip_blanks();
    while (_peek() != '}') {
        auto prop = _read_identifier();
        _skip_blanks();
        if (auto c = _peek(); c == ':') {// inline node
            _skip();
            _skip_blanks();
            auto loc = _location;
            auto impl_type = _read_identifier();
            const SceneNodeDesc *base = nullptr;
            if (_peek() == '(') { base = _parse_base_node(); }
            auto internal_node = node->define_internal(impl_type, loc, base);
            _parse_node_body(internal_node);
            node->add_property(prop, internal_node);
        } else {
            node->add_property(prop, _parse_value_list(node));
        }
        _skip_blanks();
    }
    _match('}');
}

inline SceneNodeDesc::value_list SceneParser::_parse_value_list(SceneNodeDesc *node) noexcept {
    _match('{');
    _skip_blanks();
    auto value_list = [node, this]() noexcept -> SceneNodeDesc::value_list {
        auto c = _peek();
        if (c == '}') [[unlikely]] { _report_error("Empty value list."); }
        if (c == '@' || isupper(c)) { return _parse_node_list_values(node); }
        if (c == '"' || c == '\'') { return _parse_string_list_values(); }
        if (c == 't' || c == 'f') { return _parse_bool_list_values(); }
        return _parse_number_list_values();
    }();
    _skip_blanks();
    _match('}');
    return value_list;
}

inline SceneNodeDesc::number_list SceneParser::_parse_number_list_values() noexcept {
    SceneNodeDesc::number_list list;
    list.emplace_back(_read_number());
    _skip_blanks();
    while (_peek() != '}') {
        _match(',');
        _skip_blanks();
        list.emplace_back(_read_number());
        _skip_blanks();
    }
    return list;
}

inline SceneNodeDesc::bool_list SceneParser::_parse_bool_list_values() noexcept {
    SceneNodeDesc::bool_list list;
    list.emplace_back(_read_bool());
    _skip_blanks();
    while (_peek() != '}') {
        _match(',');
        _skip_blanks();
        list.emplace_back(_read_bool());
        _skip_blanks();
    }
    return list;
}

inline SceneNodeDesc::node_list SceneParser::_parse_node_list_values(SceneNodeDesc *node) noexcept {

    auto parse_ref_or_def = [node, this]() noexcept -> const auto * {
        if (_peek() == '@') {// reference
            _skip();
            _skip_blanks();
            return _desc.reference(_read_identifier());
        }
        // inline definition
        auto loc = _location;
        auto impl_type = _read_identifier();
        const SceneNodeDesc *base = nullptr;
        if (_peek() == '(') { base = _parse_base_node(); }
        auto internal_node = node->define_internal(impl_type, loc, base);
        _parse_node_body(internal_node);
        return internal_node;
    };

    SceneNodeDesc::node_list list;
    list.emplace_back(parse_ref_or_def());
    _skip_blanks();
    while (_peek() != '}') {
        _match(',');
        _skip_blanks();
        list.emplace_back(parse_ref_or_def());
        _skip_blanks();
    }
    return list;
}

inline SceneNodeDesc::string_list SceneParser::_parse_string_list_values() noexcept {
    SceneNodeDesc::string_list list;
    list.emplace_back(_read_string());
    _skip_blanks();
    while (_peek() != '}') {
        _match(',');
        _skip_blanks();
        list.emplace_back(_read_string());
        _skip_blanks();
    }
    return list;
}

luisa::unique_ptr<SceneDesc> SceneParser::parse(
    const std::filesystem::path &entry_file, const MacroMap &cli_macros) noexcept {
    auto desc = luisa::make_unique<SceneDesc>();
    _dispatch_parse(*desc, entry_file, cli_macros);
    global_thread_pool().synchronize();
    return desc;
}

const SceneNodeDesc *SceneParser::_parse_base_node() noexcept {
    _match('(');
    _skip_blanks();
    _match('@');
    _skip_blanks();
    auto base = _desc.reference(_read_identifier());
    _skip_blanks();
    _match(')');
    return base;
}

void SceneParser::_parse_macro() noexcept {
    _skip_blanks();
    auto key = _read_identifier(true);
    if (auto cli_iter = _cli_macros.find(key);
        cli_iter != _cli_macros.end()) {
        _parsing_macros.emplace_back(cli_iter->second);
    } else if (auto local_iter = _local_macros.find(key);
               local_iter != _local_macros.end()) {
        _parsing_macros.emplace_back(local_iter->second);
    } else {
        _report_error("Undefined macro '{}'.", key);
    }
}

void SceneParser::_parse_define() noexcept {
    _skip_blanks();
    auto key = _read_identifier(true);
    _skip_blanks();
    luisa::string value;
    while (!_eof() && _peek(true) != '\n' && _peek(true) != '/') {
        value.push_back(_get(true));
    }
    if (_cli_macros.contains(key)) {
        _report_warning("Local macro '{}' is shadowed by "
                        "command-line definition.",
                        key);
    } else if (!_local_macros.insert_or_assign(key, value).second) {
        _report_warning("Macro '{}' is redefined.", key);
    }
}

}// namespace luisa::render

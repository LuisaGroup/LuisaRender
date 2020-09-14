//
// Created by Mike Smith on 2020/2/3.
//

#include "parser.h"

#include <render/filter.h>
#include <render/film.h>
#include <render/camera.h>
#include <render/shape.h>
#include <render/integrator.h>
#include <render/material.h>
#include <render/transform.h>
#include <render/task.h>
#include <render/sampler.h>

namespace luisa::render {

void Parser::_skip_blanks_and_comments() {
    LUISA_EXCEPTION_IF_NOT(_peeked.empty(), "Peeked token \"", _peeked, "\" should not be skipped at (", _curr_line, ", ", _curr_col, ")");
    while (!_remaining.empty()) {
        if (_remaining.front() == '\r') {
            _remaining = _remaining.substr(1);
            if (!_remaining.empty() && _remaining.front() == '\n') {
                _remaining = _remaining.substr(1);
            }
            _next_line++;
            _next_col = 0;
        } else if (_remaining.front() == '\n') {
            _remaining = _remaining.substr(1);
            _next_line++;
            _next_col = 0;
        } else if (std::isblank(_remaining.front())) {
            _remaining = _remaining.substr(1);
            _next_col++;
        } else if (_remaining.front() == '/') {
            _remaining = _remaining.substr(1);
            _next_col++;
            LUISA_EXCEPTION_IF(_remaining.empty() || _remaining.front() != '/', "Expected '/' at the beginning of comments at (", _next_line, ",", _next_col, ")");
            while (!_remaining.empty() && _remaining.front() != '\r' && _remaining.front() != '\n') {
                _remaining = _remaining.substr(1);
                _next_col++;
            }
        } else {
            break;
        }
    }
    _curr_line = _next_line;
    _curr_col = _next_col;
}

std::string_view Parser::_peek() {
    if (_peeked.empty()) {
        LUISA_EXCEPTION_IF(_remaining.empty(), "Peek at the end of the file at (", _curr_line, ", ", _curr_col, ")");
        if (_remaining.front() == '{' || _remaining.front() == '}' || _remaining.front() == ':' || _remaining.front() == ',' || _remaining.front() == '@') {  // symbols
            _peeked = _remaining.substr(0, 1);
            _remaining = _remaining.substr(1);
            _next_col++;
        } else if (_remaining.front() == '_' || _remaining.front() == '$' || std::isalpha(_remaining.front())) {  // Keywords or identifiers
            auto i = 1ul;
            for (; i < _remaining.size() && (_remaining[i] == '_' || std::isalnum(_remaining[i])); i++) {}
            _peeked = _remaining.substr(0, i);
            _remaining = _remaining.substr(i);
            _next_col += i;
        } else if (_remaining.front() == '+' || _remaining.front() == '-' || _remaining.front() == '.' || std::isdigit(_remaining.front())) {  // numbers
            auto i = 1ul;
            for (; i < _remaining.size() && (_remaining[i] == '.' || std::isdigit(_remaining[i])); i++) {}
            _peeked = _remaining.substr(0, i);
            _remaining = _remaining.substr(i);
            _next_col += i;
        } else if (_remaining.front() == '"') {  // strings
            auto i = 1ul;
            for (; i < _remaining.size() && _remaining[i] != '"' && _remaining[i] != '\r' && _remaining[i] != '\n'; i++) {
                if (_remaining[i] == '\\') { i++; }  // TODO: Smarter handling of escape characters
            }
            _next_col += i + 1;
            LUISA_EXCEPTION_IF(i >= _remaining.size() || _remaining[i] != '"', "Expected '\"' at (", _next_line, ", ", _next_col, ")");
            _peeked = _remaining.substr(0, i + 1);
            _remaining = _remaining.substr(i + 1);
        } else {
            LUISA_EXCEPTION("Invalid character: ", _remaining.front());
        }
    }
    return _peeked;
}

void Parser::_pop() {
    LUISA_EXCEPTION_IF(_peeked.empty(), "Token not peeked before being popped at (", _curr_line, ", ", _curr_col, ")");
    _peeked = {};
    _curr_line = _next_line;
    _curr_col = _next_col;
    _skip_blanks_and_comments();
}

void Parser::_match(std::string_view token) {
    LUISA_EXCEPTION_IF_NOT(_peek() == token, "Expected \"", token, "\", got \"", _peek(), "\" at (", _curr_line, ", ", _curr_col, ")");
}

std::shared_ptr<Task> Parser::_parse_top_level() {
    
    std::shared_ptr<Task> task;
    
    while (!_eof()) {
        auto token = _peek_and_pop();
        if (token == "task") {
            task = _parse_parameter_set()->parse<Task>();
            LUISA_WARNING_IF_NOT(_eof(), "Nodes declared after tasks will be ignored");
            break;
        }

#define LUISA_PARSER_PARSE_GLOBAL_NODE(Type)                                                                                                                    \
    if (token == #Type) {                                                                                                                                       \
        auto node_name = _peek_and_pop();                                                                                                                       \
        LUISA_EXCEPTION_IF_NOT(_is_identifier(node_name), "Invalid identifier: ", node_name);                                                                       \
        LUISA_WARNING_IF_NOT(_global_nodes.find(node_name) == _global_nodes.end(), "Duplicated global node, overwriting the one defined before: ", node_name);  \
        LUISA_INFO("Parsing global node: \"", node_name, "\", type: ", #Type);                                                                                   \
        _global_nodes.emplace(node_name, _parse_parameter_set()->parse<Type>());                                                                                \
        continue;                                                                                                                                               \
    }
    
        LUISA_PARSER_PARSE_GLOBAL_NODE(Filter)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Film)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Camera)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Shape)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Transform)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Integrator)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Material)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Task)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Sampler)
        
#undef LUISA_PARSER_PARSE_GLOBAL_NODE
    }
    
    LUISA_WARNING_IF(task == nullptr, "No tasks defined, nothing will be rendered");
    return task;
}

bool Parser::_eof() const noexcept {
    return _peeked.empty() && _remaining.empty();
}

std::shared_ptr<Task> Parser::parse(const std::filesystem::path &file_path) {
    
    _curr_line = 0;
    _curr_col = 0;
    _next_line = 0;
    _next_col = 0;
    _source.clear();
    _global_nodes.clear();
    _peeked = {};
    _remaining = {};
    _source = text_file_contents(file_path);
    _remaining = _source;
    _skip_blanks_and_comments();
    
    LUISA_INFO("Start parsing scene description file: ", file_path);
    return _parse_top_level();
}

std::unique_ptr<ParameterSet> Parser::_parse_parameter_set() {
    
    // inline creation syntax
    if (_peek() == ":") {
        _pop();  // pop ":"
        auto derived_type_name = _peek_and_pop();
        _match_and_pop("{");
        
        std::map<std::string_view, std::unique_ptr<ParameterSet>> parameters;
        while (_peek() != "}") {
            auto parameter_name = _peek_and_pop();
            LUISA_EXCEPTION_IF_NOT(_is_identifier(parameter_name), "Invalid identifier: ", parameter_name);
            LUISA_WARNING_IF_NOT(parameters.find(parameter_name) == parameters.end(), "Duplicated parameter: ", parameter_name);
            parameters.emplace(parameter_name, _parse_parameter_set());
        }
        _pop();  // pop "}"
        return std::make_unique<ParameterSet>(this, derived_type_name, std::move(parameters));
    }
    
    // value list
    std::vector<std::string_view> value_list;
    _match_and_pop("{");
    if (_peek() != "}") {
        if (_peek() == "@") {  // references
            _pop();  // pop "@"
            LUISA_EXCEPTION_IF_NOT(_is_identifier(_peek()), "Invalid reference: ", _peek());
            value_list.emplace_back(_peek_and_pop());
            while (_peek() != "}") {
                _match_and_pop(",");
                _match_and_pop("@");
                LUISA_EXCEPTION_IF_NOT(_is_identifier(_peek()), "Invalid reference: ", _peek());
                value_list.emplace_back(_peek_and_pop());
            }
        } else {  // values
            auto token = _peek_and_pop();
            if (std::isdigit(token.front()) || token.front() == '"' || token.front() == '\'') {
                value_list.emplace_back(token);
                while (_peek() != "}") {
                    _match_and_pop(",");
                    value_list.emplace_back(_peek_and_pop());
                }
            } else {  // TODO: parse inline nodes...
                LUISA_EXCEPTION("Not implemented");
            }
        }
    }
    _pop();  // pop "}"
    return std::make_unique<ParameterSet>(this, value_list);
}

std::string_view Parser::_peek_and_pop() {
    auto token = _peek();
    _pop();
    return token;
}

void Parser::_match_and_pop(std::string_view token) {
    _match(token);
    _pop();
}

bool Parser::_is_identifier(std::string_view sv) noexcept {
    if (sv.empty()) { return false; }
    if (sv.front() != '_' && sv.front() != '$' && !std::isalpha(sv.front())) { return false; }
    sv = sv.substr(1);
    return std::all_of(sv.cbegin(), sv.cend(), [](char c) noexcept { return c == '_' || c == '$' || std::isalnum(c); });
}

bool ParameterSet::_parse_bool(std::string_view sv) {
    if (sv == "true") {
        return true;
    } else if (sv == "false") {
        return false;
    }
    LUISA_EXCEPTION("Invalid bool value: ", sv);
}

float ParameterSet::_parse_float(std::string_view sv) {
    size_t offset = 0;
    auto value = std::stof(std::string{sv}, &offset);
    LUISA_EXCEPTION_IF(offset != sv.size(), "Invalid float value: ", sv);
    return value;
}

int32_t ParameterSet::_parse_int(std::string_view sv) {
    size_t offset = 0;
    auto value = std::stoi(std::string{sv}, &offset);
    LUISA_EXCEPTION_IF(offset != sv.size(), "Invalid integer value: ", sv);
    return value;
}

uint32_t ParameterSet::_parse_uint(std::string_view sv) {
    size_t offset = 0;
    auto value = std::stol(std::string{sv}, &offset);
    LUISA_EXCEPTION_IF(value < 0 || value > 0xffffffff || offset != sv.size(), "Invalid integer value: ", sv);
    return static_cast<uint32_t>(value);
}

std::string ParameterSet::_parse_string(std::string_view sv) {
    LUISA_EXCEPTION_IF(sv.size() < 2 || sv.front() != sv.back() || (sv.front() != '"' && sv.front() != '\''), "invalid string value: ", sv);
    auto raw = sv.substr(1, sv.size() - 2);
    std::string value;
    for (auto i = 0ul; i < raw.size(); i++) {
        if (raw[i] != '\'' || ++i < raw.size()) {  // TODO: Smarter handling of escape characters
            value.push_back(raw[i]);
        } else {
            LUISA_EXCEPTION("Extra escape at the end of string: ", sv);
        }
    }
    return value;
}

const ParameterSet &ParameterSet::_child(std::string_view parameter_name) const {
    auto iter = _parameters.find(parameter_name);
    if (iter == _parameters.cend()) {
        LUISA_WARNING("Parameter \"", parameter_name, "\" is not specified");
        return *_empty;
    }
    return *iter->second;
}

bool ParameterSet::parse_bool() const {
    LUISA_EXCEPTION_IF(_value_list.empty(), "No bool values given, expected exactly 1");
    LUISA_WARNING_IF(_value_list.size() != 1ul, "Too many bool values, using only the first 1");
    return _parse_bool(_value_list.front());
}

float2 ParameterSet::parse_float2() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 2, "No enough float values given, expected exactly 2");
    LUISA_WARNING_IF(_value_list.size() != 2, "Too many float values, using only the first 2");
    auto x = _parse_float(_value_list[0]);
    auto y = _parse_float(_value_list[1]);
    return make_float2(x, y);
}

std::vector<bool> ParameterSet::parse_bool_list() const {
    std::vector<bool> bool_list;
    bool_list.reserve(_value_list.size());
    for (auto sv : _value_list) { bool_list.emplace_back(_parse_bool(sv)); }
    return bool_list;
}

ParameterSet::ParameterSet(Parser *parser, std::vector<std::string_view> value_list) noexcept: _parser{parser}, _value_list{std::move(value_list)}, _is_value_list{true} {
    _empty = std::unique_ptr<ParameterSet>{new ParameterSet{parser}};
}

ParameterSet::ParameterSet(Parser *parser, std::string_view derived_type_name, std::map<std::string_view, std::unique_ptr<ParameterSet>> params) noexcept
    : _parser{parser}, _derived_type_name{derived_type_name}, _parameters{std::move(params)}, _is_value_list{false} {
    _empty = std::unique_ptr<ParameterSet>{new ParameterSet{parser}};
}

const ParameterSet &ParameterSet::operator[](std::string_view parameter_name) const {
    LUISA_INFO("Processing parameter: \"", parameter_name, "\"");
    return _child(parameter_name);
}

float ParameterSet::parse_float() const {
    LUISA_EXCEPTION_IF(_value_list.empty(), "No float values given, expected exactly 1");
    LUISA_WARNING_IF(_value_list.size() != 1, "Too many float values, using only the first 1");
    return _parse_float(_value_list.front());
}

float3 ParameterSet::parse_float3() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 3, "No enough float values given, expected exactly 3");
    LUISA_WARNING_IF(_value_list.size() != 3, "Too many float values, using only the first 3");
    auto x = _parse_float(_value_list[0]);
    auto y = _parse_float(_value_list[1]);
    auto z = _parse_float(_value_list[2]);
    return make_float3(x, y, z);
}

float4 ParameterSet::parse_float4() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 4, "No enough float values given, expected exactly 4");
    LUISA_WARNING_IF(_value_list.size() != 4, "Too many float values, using only the first 4");
    auto x = _parse_float(_value_list[0]);
    auto y = _parse_float(_value_list[1]);
    auto z = _parse_float(_value_list[2]);
    auto w = _parse_float(_value_list[3]);
    return make_float4(x, y, z, w);
}

float3x3 ParameterSet::parse_float3x3() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 9, "No enough float values given, expected exactly 16");
    LUISA_WARNING_IF(_value_list.size() != 9, "Too many float values, using only the first 16");
    auto m00 = _parse_float(_value_list[0]);
    auto m01 = _parse_float(_value_list[1]);
    auto m02 = _parse_float(_value_list[2]);
    auto m10 = _parse_float(_value_list[3]);
    auto m11 = _parse_float(_value_list[4]);
    auto m12 = _parse_float(_value_list[5]);
    auto m20 = _parse_float(_value_list[6]);
    auto m21 = _parse_float(_value_list[7]);
    auto m22 = _parse_float(_value_list[8]);
    return make_float3x3(m00, m01, m02, m10, m11, m12, m20, m21, m22);
}

float4x4 ParameterSet::parse_float4x4() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 16, "No enough float values given, expected exactly 16");
    LUISA_WARNING_IF(_value_list.size() != 16, "Too many float values, using only the first 16");
    auto m00 = _parse_float(_value_list[0]);
    auto m01 = _parse_float(_value_list[1]);
    auto m02 = _parse_float(_value_list[2]);
    auto m03 = _parse_float(_value_list[3]);
    auto m10 = _parse_float(_value_list[4]);
    auto m11 = _parse_float(_value_list[5]);
    auto m12 = _parse_float(_value_list[6]);
    auto m13 = _parse_float(_value_list[7]);
    auto m20 = _parse_float(_value_list[8]);
    auto m21 = _parse_float(_value_list[9]);
    auto m22 = _parse_float(_value_list[10]);
    auto m23 = _parse_float(_value_list[11]);
    auto m30 = _parse_float(_value_list[12]);
    auto m31 = _parse_float(_value_list[13]);
    auto m32 = _parse_float(_value_list[14]);
    auto m33 = _parse_float(_value_list[15]);
    return make_float4x4(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);
}

std::vector<float> ParameterSet::parse_float_list() const {
    std::vector<float> float_list;
    float_list.reserve(_value_list.size());
    for (auto sv : _value_list) {
        float_list.emplace_back(_parse_float(sv));
    }
    return float_list;
}

int ParameterSet::parse_int() const {
    LUISA_EXCEPTION_IF(_value_list.empty(), "No int values given, expected exactly 1");
    LUISA_WARNING_IF(_value_list.size() != 1, "Too many int values, using only the first 1");
    return _parse_int(_value_list.front());
}

int2 ParameterSet::parse_int2() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 2, "No enough int values given, expected exactly 2");
    LUISA_WARNING_IF(_value_list.size() != 2, "Too many int values, using only the first 2");
    auto x = _parse_int(_value_list[0]);
    auto y = _parse_int(_value_list[1]);
    return make_int2(x, y);
}

int3 ParameterSet::parse_int3() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 3, "No enough int values given, expected exactly 3");
    LUISA_WARNING_IF(_value_list.size() != 3, "Too many int values, using only the first 3");
    auto x = _parse_int(_value_list[0]);
    auto y = _parse_int(_value_list[1]);
    auto z = _parse_int(_value_list[2]);
    return make_int3(x, y, z);
}

int4 ParameterSet::parse_int4() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 4, "No enough int values given, expected exactly 4");
    LUISA_WARNING_IF(_value_list.size() != 4, "Too many int values, using only the first 4");
    auto x = _parse_int(_value_list[0]);
    auto y = _parse_int(_value_list[1]);
    auto z = _parse_int(_value_list[2]);
    auto w = _parse_int(_value_list[3]);
    return make_int4(x, y, z, w);
}

std::vector<int32_t> ParameterSet::parse_int_list() const {
    std::vector<int32_t> int_list;
    int_list.reserve(_value_list.size());
    for (auto sv : _value_list) {
        int_list.emplace_back(_parse_int(sv));
    }
    return int_list;
}

uint ParameterSet::parse_uint() const {
    LUISA_EXCEPTION_IF(_value_list.empty(), "No uint values given, expected exactly 1");
    LUISA_WARNING_IF(_value_list.size() != 1, "Too many uint values, using only the first 1");
    return _parse_uint(_value_list.front());
}

uint2 ParameterSet::parse_uint2() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 2, "No enough uint values given, expected exactly 2");
    LUISA_WARNING_IF(_value_list.size() != 2, "Too many uint values, using only the first 2");
    auto x = _parse_uint(_value_list[0]);
    auto y = _parse_uint(_value_list[1]);
    return make_uint2(x, y);
}

uint3 ParameterSet::parse_uint3() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 3, "No enough uint values given, expected exactly 3");
    LUISA_WARNING_IF(_value_list.size() != 3, "Too many uint values, using only the first 3");
    auto x = _parse_uint(_value_list[0]);
    auto y = _parse_uint(_value_list[1]);
    auto z = _parse_uint(_value_list[2]);
    return make_uint3(x, y, z);
}

uint4 ParameterSet::parse_uint4() const {
    LUISA_EXCEPTION_IF(_value_list.size() < 4, "No enough uint values given, expected exactly 4");
    LUISA_WARNING_IF(_value_list.size() != 4, "Too many uint values, using only the first 4");
    auto x = _parse_uint(_value_list[0]);
    auto y = _parse_uint(_value_list[1]);
    auto z = _parse_uint(_value_list[2]);
    auto w = _parse_uint(_value_list[3]);
    return make_uint4(x, y, z, w);
}

std::vector<uint32_t> ParameterSet::parse_uint_list() const {
    std::vector<uint32_t> uint_list;
    uint_list.reserve(_value_list.size());
    for (auto sv : _value_list) {
        uint_list.emplace_back(_parse_uint(sv));
    }
    return uint_list;
}

std::string ParameterSet::parse_string() const {
    LUISA_EXCEPTION_IF(_value_list.empty(), "No string values given, expected exactly 1");
    LUISA_WARNING_IF(_value_list.size() != 1, "Too many uint values, using only the first 1");
    return _parse_string(_value_list.front());
}

std::string ParameterSet::parse_string_or_default(std::string_view default_value) const noexcept {
    try { return parse_string(); } catch (const std::runtime_error &e) {
        LUISA_WARNING("Error occurred while parsing parameter, using default value: \"", default_value, "\"");
        return std::string{default_value};
    }
}

std::vector<std::string> ParameterSet::parse_string_list() const {
    std::vector<std::string> string_list;
    string_list.reserve(_value_list.size());
    for (auto sv : _value_list) {
        string_list.emplace_back(_parse_string(sv));
    }
    return string_list;
}

}

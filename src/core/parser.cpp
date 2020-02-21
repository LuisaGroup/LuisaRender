//
// Created by Mike Smith on 2020/2/3.
//

#include "parser.h"

#include "filter.h"
#include "film.h"
#include "camera.h"
#include "shape.h"
#include "integrator.h"
#include "material.h"
#include "transform.h"
#include "render.h"
#include "sampler.h"

namespace luisa {

void Parser::_skip_blanks_and_comments() {
    LUISA_ERROR_IF_NOT(_peeked.empty(), "peeked token \"", _peeked, "\" should not be skipped at (", _curr_line, ", ", _curr_col, ")");
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
            LUISA_ERROR_IF(_remaining.empty() || _remaining.front() != '/', "expected '/' at the beginning of comments at (", _next_line, ",", _next_col, ")");
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
        LUISA_ERROR_IF(_remaining.empty(), "peek at the end of the file at (", _curr_line, ", ", _curr_col, ")");
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
            LUISA_ERROR_IF(i >= _remaining.size() || _remaining[i] != '"', "expected '\"' at (", _next_line, ", ", _next_col, ")");
            _peeked = _remaining.substr(0, i + 1);
            _remaining = _remaining.substr(i + 1);
        } else {
            LUISA_ERROR("invalid character: ", _remaining.front());
        }
    }
    return _peeked;
}

void Parser::_pop() {
    LUISA_ERROR_IF(_peeked.empty(), "token not peeked before being popped at (", _curr_line, ", ", _curr_col, ")");
    _peeked = {};
    _curr_line = _next_line;
    _curr_col = _next_col;
    _skip_blanks_and_comments();
}

void Parser::_match(std::string_view token) {
    LUISA_ERROR_IF_NOT(_peek() == token, "expected \"", token, "\", got \"", _peek(), "\" at (", _curr_line, ", ", _curr_col, ")");
}

std::shared_ptr<Render> Parser::_parse_top_level() {
    
    std::shared_ptr<Render> task;
    
    while (!_eof()) {
        auto token = _peek_and_pop();
        if (token == "renderer") {
            task = _parse_parameter_set()->parse<Render>();
            LUISA_WARNING_IF_NOT(_eof(), "nodes declared after tasks will be ignored");
            break;
        }

#define LUISA_PARSER_PARSE_GLOBAL_NODE(Type)                                                                                                                    \
    if (token == #Type) {                                                                                                                                       \
        auto node_name = _peek_and_pop();                                                                                                                       \
        LUISA_ERROR_IF_NOT(_is_identifier(node_name), "invalid identifier: ", node_name);                                                                       \
        LUISA_WARNING_IF_NOT(_global_nodes.find(node_name) == _global_nodes.end(), "duplicated global node, overwriting the one defined before: ", node_name);  \
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
        LUISA_PARSER_PARSE_GLOBAL_NODE(Render)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Sampler)
        LUISA_PARSER_PARSE_GLOBAL_NODE(Light)
        
#undef LUISA_PARSER_PARSE_GLOBAL_NODE
    }
    
    LUISA_WARNING_IF(task == nullptr, "no tasks defined, nothing will be rendered");
    return task;
}

bool Parser::_eof() const noexcept {
    return _peeked.empty() && _remaining.empty();
}

std::shared_ptr<Render> Parser::parse(const std::filesystem::path &file_path) {
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
            LUISA_ERROR_IF_NOT(_is_identifier(parameter_name), "invalid identifier: ", parameter_name);
            LUISA_WARNING_IF_NOT(parameters.find(parameter_name) == parameters.end(), "duplicated parameter: ", parameter_name);
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
            LUISA_ERROR_IF_NOT(_is_identifier(_peek()), "invalid reference: ", _peek());
            value_list.emplace_back(_peek_and_pop());
            while (_peek() != "}") {
                _match_and_pop(",");
                _match_and_pop("@");
                LUISA_ERROR_IF_NOT(_is_identifier(_peek()), "invalid reference: ", _peek());
                value_list.emplace_back(_peek_and_pop());
            }
        } else {  // values
            value_list.emplace_back(_peek_and_pop());
            while (_peek() != "}") {
                _match_and_pop(",");
                value_list.emplace_back(_peek_and_pop());
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

}

//
// Created by Mike Smith on 2020/2/3.
//

#pragma once

#include <string_view>
#include <map>
#include <unordered_map>
#include <vector>

#include <util/logging.h>

#include "data_types.h"
#include "node.h"

namespace luisa {

class Parser;

class ParameterSet {

private:
    Parser *_parser;
    std::string_view _derived_type_name;
    std::vector<std::string_view> _value_list;
    std::map<std::string_view, std::unique_ptr<ParameterSet>> _parameters;
    bool _is_value_list{};
    
    [[nodiscard]] static bool _parse_bool(std::string_view sv) {
        if (sv == "true") {
            return true;
        } else if (sv == "false") {
            return false;
        }
        LUISA_EXCEPTION("Invalid bool value: ", sv);
    }
    
    [[nodiscard]] static float _parse_float(std::string_view sv) {
        size_t offset = 0;
        auto value = std::stof(std::string{sv}, &offset);
        LUISA_EXCEPTION_IF(offset != sv.size(), "Invalid float value: ", sv);
        return value;
    }
    
    [[nodiscard]] static int32_t _parse_int(std::string_view sv) {
        size_t offset = 0;
        auto value = std::stoi(std::string{sv}, &offset);
        LUISA_EXCEPTION_IF(offset != sv.size(), "Invalid integer value: ", sv);
        return value;
    }
    
    [[nodiscard]] static uint32_t _parse_uint(std::string_view sv) {
        size_t offset = 0;
        auto value = std::stol(std::string{sv}, &offset);
        LUISA_EXCEPTION_IF(value < 0 || value > 0xffffffff || offset != sv.size(), "Invalid integer value: ", sv);
        return static_cast<uint32_t>(value);
    }
    
    [[nodiscard]] static std::string _parse_string(std::string_view sv) {
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
    
    [[nodiscard]] const ParameterSet &_child(std::string_view parameter_name) const {
        auto iter = _parameters.find(parameter_name);
        if (iter == _parameters.cend()) {
            LUISA_WARNING("Parameter \"", parameter_name, "\" is not specific");
            return *_empty;
        }
        return *iter->second;
    }

private:
    std::unique_ptr<ParameterSet> _empty{};
    ParameterSet(Parser *parser) noexcept: _parser{parser} {}  // only for making _empty

public:
    ParameterSet(Parser *parser, std::vector<std::string_view> value_list) noexcept: _parser{parser}, _value_list{std::move(value_list)}, _is_value_list{true} {
        _empty = std::unique_ptr<ParameterSet>{new ParameterSet{parser}};
    }
    
    ParameterSet(Parser *parser, std::string_view derived_type_name, std::map<std::string_view, std::unique_ptr<ParameterSet>> params) noexcept
        : _parser{parser}, _derived_type_name{derived_type_name}, _parameters{std::move(params)}, _is_value_list{false} {
        _empty = std::unique_ptr<ParameterSet>{new ParameterSet{parser}};
    }
    
    [[nodiscard]] const ParameterSet &operator[](std::string_view parameter_name) const {
        LUISA_INFO("Processing parameter \"", parameter_name, "\"");
        return _child(parameter_name);
    }
    
    template<typename BaseClass>
    [[nodiscard]] std::shared_ptr<BaseClass> parse() const;
    
    template<typename BaseClass>
    [[nodiscard]] auto parse_or_null() const {
        std::shared_ptr<BaseClass> p = nullptr;
        try {
            p = parse<BaseClass>();
        } catch (const std::runtime_error &e) {
            LUISA_WARNING("Error occurred while parsing parameter, returning null");
        }
        return p;
    }
    
    template<typename BaseClass>
    [[nodiscard]] std::vector<std::shared_ptr<BaseClass>> parse_reference_list() const;
    
    [[nodiscard]] bool parse_bool() const {
        LUISA_EXCEPTION_IF(_value_list.empty(), "No bool values given, expected exactly 1");
        LUISA_WARNING_IF(_value_list.size() != 1ul, "Too many bool values, using only the first 1");
        return _parse_bool(_value_list.front());
    }
    
    [[nodiscard]] std::vector<bool> parse_bool_list() const {
        std::vector<bool> bool_list;
        bool_list.reserve(_value_list.size());
        for (auto sv : _value_list) {
            bool_list.emplace_back(_parse_bool(sv));
        }
        return bool_list;
    }
    
    [[nodiscard]] float parse_float() const {
        LUISA_EXCEPTION_IF(_value_list.empty(), "No float values given, expected exactly 1");
        LUISA_WARNING_IF(_value_list.size() != 1, "Too many float values, using only the first 1");
        return _parse_float(_value_list.front());
    }
    
    [[nodiscard]] float2 parse_float2() const {
        LUISA_EXCEPTION_IF(_value_list.size() < 2, "No enough float values given, expected exactly 2");
        LUISA_WARNING_IF(_value_list.size() != 2, "Too many float values, using only the first 2");
        auto x = _parse_float(_value_list[0]);
        auto y = _parse_float(_value_list[1]);
        return make_float2(x, y);
    }
    
    [[nodiscard]] float3 parse_float3() const {
        LUISA_EXCEPTION_IF(_value_list.size() < 3, "No enough float values given, expected exactly 3");
        LUISA_WARNING_IF(_value_list.size() != 3, "Too many float values, using only the first 3");
        auto x = _parse_float(_value_list[0]);
        auto y = _parse_float(_value_list[1]);
        auto z = _parse_float(_value_list[2]);
        return make_float3(x, y, z);
    }
    
    [[nodiscard]] float4 parse_float4() const {
        LUISA_EXCEPTION_IF(_value_list.size() < 4, "No enough float values given, expected exactly 4");
        LUISA_WARNING_IF(_value_list.size() != 4, "Too many float values, using only the first 4");
        auto x = _parse_float(_value_list[0]);
        auto y = _parse_float(_value_list[1]);
        auto z = _parse_float(_value_list[2]);
        auto w = _parse_float(_value_list[3]);
        return make_float4(x, y, z, w);
    }
    
    [[nodiscard]] float3x3 parse_float3x3() const {
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
    
    [[nodiscard]] float4x4 parse_float4x4() const {
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
    
    [[nodiscard]] std::vector<float> parse_float_list() const {
        std::vector<float> float_list;
        float_list.reserve(_value_list.size());
        for (auto sv : _value_list) {
            float_list.emplace_back(_parse_float(sv));
        }
        return float_list;
    }
    
    [[nodiscard]] int parse_int() const {
        LUISA_EXCEPTION_IF(_value_list.empty(), "No int values given, expected exactly 1");
        LUISA_WARNING_IF(_value_list.size() != 1, "Too many int values, using only the first 1");
        return _parse_int(_value_list.front());
    }
    
    [[nodiscard]] int2 parse_int2() const {
        LUISA_EXCEPTION_IF(_value_list.size() < 2, "No enough int values given, expected exactly 2");
        LUISA_WARNING_IF(_value_list.size() != 2, "Too many int values, using only the first 2");
        auto x = _parse_int(_value_list[0]);
        auto y = _parse_int(_value_list[1]);
        return make_int2(x, y);
    }
    
    [[nodiscard]] int3 parse_int3() const {
        LUISA_EXCEPTION_IF(_value_list.size() < 3, "No enough int values given, expected exactly 3");
        LUISA_WARNING_IF(_value_list.size() != 3, "Too many int values, using only the first 3");
        auto x = _parse_int(_value_list[0]);
        auto y = _parse_int(_value_list[1]);
        auto z = _parse_int(_value_list[2]);
        return make_int3(x, y, z);
    }
    
    [[nodiscard]] int4 parse_int4() const {
        
        LUISA_EXCEPTION_IF(_value_list.size() < 4, "No enough int values given, expected exactly 4");
        LUISA_WARNING_IF(_value_list.size() != 4, "Too many int values, using only the first 4");
        auto x = _parse_int(_value_list[0]);
        auto y = _parse_int(_value_list[1]);
        auto z = _parse_int(_value_list[2]);
        auto w = _parse_int(_value_list[3]);
        return make_int4(x, y, z, w);
    }
    
    [[nodiscard]] std::vector<int32_t> parse_int_list() const {
        std::vector<int32_t> int_list;
        int_list.reserve(_value_list.size());
        for (auto sv : _value_list) {
            int_list.emplace_back(_parse_int(sv));
        }
        return int_list;
    }
    
    [[nodiscard]] uint parse_uint() const {
        LUISA_EXCEPTION_IF(_value_list.empty(), "No uint values given, expected exactly 1");
        LUISA_WARNING_IF(_value_list.size() != 1, "Too many uint values, using only the first 1");
        return _parse_uint(_value_list.front());
    }
    
    [[nodiscard]] uint2 parse_uint2() const {
        LUISA_EXCEPTION_IF(_value_list.size() < 2, "No enough uint values given, expected exactly 2");
        LUISA_WARNING_IF(_value_list.size() != 2, "Too many uint values, using only the first 2");
        auto x = _parse_uint(_value_list[0]);
        auto y = _parse_uint(_value_list[1]);
        return make_uint2(x, y);
    }
    
    [[nodiscard]] uint3 parse_uint3() const {
        LUISA_EXCEPTION_IF(_value_list.size() < 3, "No enough uint values given, expected exactly 3");
        LUISA_WARNING_IF(_value_list.size() != 3, "Too many uint values, using only the first 3");
        auto x = _parse_uint(_value_list[0]);
        auto y = _parse_uint(_value_list[1]);
        auto z = _parse_uint(_value_list[2]);
        return make_uint3(x, y, z);
    }
    
    [[nodiscard]] uint4 parse_uint4() const {
        LUISA_EXCEPTION_IF(_value_list.size() < 4, "No enough uint values given, expected exactly 4");
        LUISA_WARNING_IF(_value_list.size() != 4, "Too many uint values, using only the first 4");
        auto x = _parse_uint(_value_list[0]);
        auto y = _parse_uint(_value_list[1]);
        auto z = _parse_uint(_value_list[2]);
        auto w = _parse_uint(_value_list[3]);
        return make_uint4(x, y, z, w);
    }
    
    [[nodiscard]] std::vector<uint32_t> parse_uint_list() const {
        std::vector<uint32_t> uint_list;
        uint_list.reserve(_value_list.size());
        for (auto sv : _value_list) {
            uint_list.emplace_back(_parse_uint(sv));
        }
        return uint_list;
    }
    
    [[nodiscard]] std::string parse_string() const {
        LUISA_EXCEPTION_IF(_value_list.empty(), "No uint values given, expected exactly 1");
        LUISA_WARNING_IF(_value_list.size() != 1, "Too many uint values, using only the first 1");
        return _parse_string(_value_list.front());
    }
    
    [[nodiscard]] std::string parse_string_or_default(std::string_view default_value) const noexcept {
        try { return parse_string(); } catch (const std::runtime_error &e) {
            LUISA_WARNING("Error occurred while parsing parameter, using default value: \"", default_value, "\"");
            return std::string{default_value};
        }
    }
    
    [[nodiscard]] std::vector<std::string> parse_string_list() const {
        std::vector<std::string> string_list;
        string_list.reserve(_value_list.size());
        for (auto sv : _value_list) {
            string_list.emplace_back(_parse_string(sv));
        }
        return string_list;
    }

#define LUISA_PARAMETER_SET_PARSE_OR_DEFAULT(Type, TO_STRING)                                                        \
    [[nodiscard]] Type parse_##Type##_or_default(Type default_value) const noexcept {                                \
        try { return parse_##Type(); } catch (const std::runtime_error &e) {                                         \
            LUISA_WARNING("Error occurred while parsing parameter, using default value: ", TO_STRING(default_value));  \
            return default_value;                                                                                    \
        }                                                                                                            \
    }

#define LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_SCALAR(Type)  \
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT(Type, std::to_string)

#define LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(Type)  \
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT(Type, glm::to_string)
    
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT(bool, [](bool v) noexcept { return v ? "true" : "false"; })
    
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_SCALAR(float)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_SCALAR(int)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_SCALAR(uint)
    
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(float2)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(float3)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(float4)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(float3x3)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(float4x4)
    
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(int2)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(int3)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(int4)
    
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(uint2)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(uint3)
    LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR(uint4)

#undef LUISA_PARAMETER_SET_PARSE_OR_DEFAULT
#undef LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_SCALAR
#undef LUISA_PARAMETER_SET_PARSE_OR_DEFAULT_VECTOR

};

class Render;

class Parser {

private:
    Device *_device;
    size_t _curr_line{};
    size_t _curr_col{};
    size_t _next_line{};
    size_t _next_col{};
    std::string _source;
    std::string_view _peeked;
    std::string_view _remaining;
    std::unordered_map<std::string_view, std::shared_ptr<Node>> _global_nodes;
    
    void _skip_blanks_and_comments();
    void _pop();
    void _match(std::string_view token);
    void _match_and_pop(std::string_view token);
    [[nodiscard]] static bool _is_identifier(std::string_view sv) noexcept;
    [[nodiscard]] std::string_view _peek();
    [[nodiscard]] std::string_view _peek_and_pop();
    [[nodiscard]] std::shared_ptr<Render> _parse_top_level();
    [[nodiscard]] bool _eof() const noexcept;
    [[nodiscard]] std::unique_ptr<ParameterSet> _parse_parameter_set();

public:
    explicit Parser(Device *device) noexcept: _device{device} {}
    
    [[nodiscard]] std::shared_ptr<Render> parse(const std::filesystem::path &file_path);
    
    template<typename T>
    [[nodiscard]] std::shared_ptr<T> global_node(std::string_view node_name) const {
        if (auto iter = _global_nodes.find(node_name); iter != _global_nodes.end()) {
            if (auto p = std::dynamic_pointer_cast<T>(iter->second)) { return p; }
            LUISA_EXCEPTION("Incompatible type for node: ", node_name);
        }
        LUISA_EXCEPTION("Undefined node: ", node_name);
    }
    
    [[nodiscard]] Device &device() noexcept { return *_device; }
};

template<typename BaseClass>
[[nodiscard]] std::shared_ptr<BaseClass> ParameterSet::parse() const {
    if (_is_value_list) {
        LUISA_WARNING_IF_NOT(_value_list.size() == 1, "Too many references given, using only the first 1");
        return _parser->global_node<BaseClass>(_value_list.front());
    }
    return BaseClass::_creators[_derived_type_name](&_parser->device(), *this);
}

template<typename BaseClass>
std::vector<std::shared_ptr<BaseClass>> ParameterSet::parse_reference_list() const {
    std::vector<std::shared_ptr<BaseClass>> node_list;
    for (auto sv : _value_list) {
        node_list.emplace_back(_parser->global_node<BaseClass>(sv));
    }
    return node_list;
}

}

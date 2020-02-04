//
// Created by Mike Smith on 2020/2/3.
//

#pragma once

#include <string_view>
#include <map>
#include <unordered_map>
#include <vector>

#include <util/exception.h>

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
    
    [[nodiscard]] static bool _parse_bool(std::string_view sv) {
        if (sv == "true") {
            return true;
        } else if (sv == "false") {
            return false;
        }
        LUISA_ERROR("invalid bool value: ", sv);
    }
    
    [[nodiscard]] static float _parse_float(std::string_view sv) {
        size_t offset = 0;
        auto value = std::stof(std::string{sv}, &offset);
        LUISA_ERROR_IF(offset != sv.size(), "invalid float value: ", sv);
        return value;
    }
    
    [[nodiscard]] static int32_t _parse_integer(std::string_view sv) {
        size_t offset = 0;
        auto value = std::stoi(std::string{sv}, &offset);
        LUISA_ERROR_IF(offset != sv.size(), "invalid integer value: ", sv);
        return value;
    }
    
    [[nodiscard]] static std::string _parse_string(std::string_view sv) {
        LUISA_ERROR_IF(sv.size() < 2 || sv.front() != sv.back() || (sv.front() != '"' && sv.front() != '\''), "invalid string value: ", sv);
        auto raw = sv.substr(1, sv.size() - 2);
        std::string value;
        for (auto i = 0ul; i < raw.size(); i++) {
            if (raw[i] != '\'' || ++i < raw.size()) {  // TODO: Smarter handling of escape characters
                value.push_back(raw[i]);
            } else {
                LUISA_ERROR("extra escape at the end of string: ", sv);
            }
        }
        return value;
    }

public:
    template<typename BaseClass>
    [[nodiscard]] std::shared_ptr<BaseClass> parse_node(std::string_view parameter_name) const;
    
    template<typename BaseClass>
    [[nodiscard]] std::shared_ptr<BaseClass> parse_node_or_default(std::string_view parameter_name, const std::shared_ptr<BaseClass> &default_node) noexcept {
        try { return parse_node<BaseClass>(parameter_name); } catch (const std::runtime_error &e) {
            LUISA_WARNING("error occurred while parsing node: ", e.what());
            return default_node;
        }
    }
    
    template<typename BaseClass>
    [[nodiscard]] std::vector<std::shared_ptr<BaseClass>> parse_node_list() const;
    
    [[nodiscard]] bool parse_bool() const {
        LUISA_ERROR_IF(_value_list.empty(), "no bool values given, expected exactly one");
        LUISA_WARNING_IF(_value_list.size() != 1ul, "too many bool values, using only the first one");
        return _parse_bool(_value_list.front());
    }
    
    [[nodiscard]] bool parse_bool_or_default(bool default_value) const {
        try { return parse_bool(); } catch (const std::runtime_error &e) {
            LUISA_WARNING("error occurred while parsing bool: ", e.what());
            return default_value;
        }
    }
    
    [[nodiscard]] std::vector<bool> parse_bool_list() const {
        std::vector<bool> bool_list;
        for (auto sv : _value_list) { bool_list.emplace_back(_parse_bool(sv)); }
        return bool_list;
    }
    
    [[nodiscard]] float parse_float() const {
        LUISA_ERROR_IF(_value_list.empty(), "no float values given, expected exactly one");
        LUISA_WARNING_IF(_value_list.size() != 1, "too many float values, using only the first one");
        return _parse_float(_value_list.front());
    }
    
    [[nodiscard]] float2 parse_float2() const {
        LUISA_ERROR_IF(_value_list.size() < 2, "no enough float values given, expected exactly two");
        LUISA_WARNING_IF(_value_list.size() != 2, "too many float values, using only the first two");
        auto x = _parse_float(_value_list[0]);
        auto y = _parse_float(_value_list[1]);
        return make_float2(x, y);
    }
    
    [[nodiscard]] float3 parse_float3() const {
        LUISA_ERROR_IF(_value_list.size() < 3, "no enough float values given, expected exactly three");
        LUISA_WARNING_IF(_value_list.size() != 3, "too many float values, using only the first three");
        auto x = _parse_float(_value_list[0]);
        auto y = _parse_float(_value_list[1]);
        auto z = _parse_float(_value_list[2]);
        return make_float3(x, y, z);
    }
    
    [[nodiscard]] float4 parse_float4() const {
        LUISA_ERROR_IF(_value_list.size() < 4, "no enough float values given, expected exactly four");
        LUISA_WARNING_IF(_value_list.size() != 4, "too many float values, using only the first four");
        auto x = _parse_float(_value_list[0]);
        auto y = _parse_float(_value_list[1]);
        auto z = _parse_float(_value_list[2]);
        auto w = _parse_float(_value_list[3]);
        return make_float4(x, y, z, w);
    }
    
    [[nodiscard]] std::vector<float> parse_float_list() const {
        std::vector<float> float_list;
        for (auto sv : _value_list) { float_list.emplace_back(_parse_float(sv)); }
        return float_list;
    }
    
    [[nodiscard]] std::vector<int32_t> parse_integer_list() const {
        std::vector<int32_t> integer_list;
        for (auto sv : _value_list) { integer_list.emplace_back(_parse_integer(sv)); }
        return integer_list;
    }
    
    [[nodiscard]] std::vector<std::string> parse_string_list() const {
        std::vector<std::string> string_list;
        for (auto sv : _value_list) { string_list.emplace_back(_parse_string(sv)); }
        return string_list;
    }
    
};

class Parser {

private:
    Device *_device;
    std::unordered_map<std::string, std::shared_ptr<Node>> _global_nodes;

public:
    template<typename T>
    [[nodiscard]] const std::shared_ptr<T> &global_node(const std::string &node_name) const noexcept {
        if (auto iter = _global_nodes.find(node_name); iter != _global_nodes.end()) {
            if (auto p = std::dynamic_pointer_cast<T>(iter->second)) { return p; }
            LUISA_ERROR("incompatible type for node: ", node_name);
        }
        LUISA_ERROR("undefined node: ", node_name);
    }
    [[nodiscard]] Device &device() noexcept { return *_device; }
};

template<typename BaseClass>
std::shared_ptr<BaseClass> ParameterSet::parse_node(std::string_view parameter_name) const {
    if (auto iter = _parameters.find(parameter_name); iter != _parameters.end()) {
        return BaseClass::creators[iter->second->_derived_type_name](&_parser->device(), *iter->second);
    }
    LUISA_ERROR("undefined parameter: ", parameter_name);
}

template<typename BaseClass>
std::vector<std::shared_ptr<BaseClass>> ParameterSet::parse_node_list() const {
    std::vector<std::shared_ptr<BaseClass>> node_list;
    for (auto sv : _value_list) {
        LUISA_ERROR_IF(sv.empty() || sv.front() != '@', "invalid reference: ", sv);
        do { sv = sv.substr(1); } while (!sv.empty() && std::isblank(sv.front()));
        node_list.emplace_back(_parser->global_node<BaseClass>(sv));
    }
    return node_list;
}

}

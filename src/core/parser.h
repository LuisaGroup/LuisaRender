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

public:
    template<typename BaseClass>
    [[nodiscard]] std::unique_ptr<BaseClass> parse_node(std::string_view parameter_name) const;
    
    template<typename BaseClass>
    [[nodiscard]] std::vector<std::shared_ptr<BaseClass>> parse_node_list() const;
    
    [[nodiscard]] std::vector<bool> parse_bool_list() const {
        std::vector<bool> bool_list;
        for (auto sv : _value_list) {
            if (sv == "true") {
                bool_list.emplace_back(true);
            } else if (sv == "false") {
                bool_list.emplace_back(false);
            } else {
                LUISA_ERROR("invalid bool value: ", sv);
            }
        }
        return bool_list;
    }
    
    [[nodiscard]] std::vector<float> parse_float_list() const {
        std::vector<float> float_list;
        for (auto sv : _value_list) {
            size_t offset = 0;
            auto value = std::stof(std::string{sv}, &offset);
            LUISA_ERROR_IF(offset != sv.size(), "invalid float value: ", sv);
            float_list.emplace_back(value);
        }
        return float_list;
    }
    
    [[nodiscard]] std::vector<int32_t> parse_integer_list() const {
        std::vector<int32_t> integer_list;
        for (auto sv : _value_list) {
            size_t offset = 0;
            auto value = std::stoi(std::string{sv}, &offset);
            LUISA_ERROR_IF(offset != sv.size(), "invalid integer value: ", sv);
            integer_list.emplace_back(value);
        }
        return integer_list;
    }
    
    [[nodiscard]] std::vector<std::string> parse_string_list() const {
        std::vector<std::string> string_list;
        for (auto sv : _value_list) {
            LUISA_ERROR_IF(sv.size() < 2 || sv.front() != sv.back() || (sv.front() != '"' && sv.front() != '\''), "invalid string value: ", sv);
            auto raw = sv.substr(1, sv.size() - 2);
            std::string escaped;
            for (auto i = 0ul; i < raw.size(); i++) {
                if (raw[i] != '\'' || ++i < raw.size()) {  // TODO: Smarter handling of escape characters
                    escaped.push_back(raw[i]);
                } else {
                    LUISA_ERROR("extra escape at the end of string: ", sv);
                }
            }
            string_list.emplace_back(escaped);
        }
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
std::unique_ptr<BaseClass> ParameterSet::parse_node(std::string_view parameter_name) const {
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

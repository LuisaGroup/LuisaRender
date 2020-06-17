//
// Created by Mike Smith on 2020/2/3.
//

#pragma once

#include <string_view>
#include <map>
#include <vector>
#include <memory>

#include <util/logging.h>

#include "data_types.h"
#include "plugin.h"

namespace luisa {

class Parser;

class ParameterSet {

private:
    Parser *_parser;
    std::string_view _derived_type_name;
    std::vector<std::string_view> _value_list;
    std::map<std::string_view, std::unique_ptr<ParameterSet>> _parameters;
    bool _is_value_list{};
    
    [[nodiscard]] static bool _parse_bool(std::string_view sv);
    [[nodiscard]] static float _parse_float(std::string_view sv);
    [[nodiscard]] static int32_t _parse_int(std::string_view sv);
    [[nodiscard]] static uint32_t _parse_uint(std::string_view sv);
    [[nodiscard]] static std::string _parse_string(std::string_view sv);
    [[nodiscard]] const ParameterSet &_child(std::string_view parameter_name) const;

private:
    std::unique_ptr<ParameterSet> _empty{};
    explicit ParameterSet(Parser *parser) noexcept: _parser{parser} {}  // only for making _empty

public:
    ParameterSet(Parser *parser, std::vector<std::string_view> value_list) noexcept;
    ParameterSet(Parser *parser, std::string_view derived_type_name, std::map<std::string_view, std::unique_ptr<ParameterSet>> params) noexcept;
    [[nodiscard]] const ParameterSet &operator[](std::string_view parameter_name) const;
    
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
    
    [[nodiscard]] bool parse_bool() const;
    [[nodiscard]] std::vector<bool> parse_bool_list() const;
    [[nodiscard]] float parse_float() const;
    [[nodiscard]] float2 parse_float2() const;
    [[nodiscard]] float3 parse_float3() const;
    [[nodiscard]] float4 parse_float4() const;
    [[nodiscard]] float3x3 parse_float3x3() const;
    [[nodiscard]] float4x4 parse_float4x4() const;
    [[nodiscard]] std::vector<float> parse_float_list() const;
    [[nodiscard]] int parse_int() const;
    [[nodiscard]] int2 parse_int2() const;
    [[nodiscard]] int3 parse_int3() const;
    [[nodiscard]] int4 parse_int4() const;
    [[nodiscard]] std::vector<int32_t> parse_int_list() const;
    [[nodiscard]] uint parse_uint() const;
    [[nodiscard]] uint2 parse_uint2() const;
    [[nodiscard]] uint3 parse_uint3() const;
    [[nodiscard]] uint4 parse_uint4() const;
    [[nodiscard]] std::vector<uint32_t> parse_uint_list() const;
    [[nodiscard]] std::string parse_string() const;
    [[nodiscard]] std::string parse_string_or_default(std::string_view default_value) const noexcept;
    [[nodiscard]] std::vector<std::string> parse_string_list() const;

#define LUISA_PARAMETER_SET_PARSE_OR_DEFAULT(Type, TO_STRING)                                                          \
    [[nodiscard]] Type parse_##Type##_or_default(Type default_value) const noexcept {                                  \
        try { return parse_##Type(); } catch (const std::runtime_error &e) {                                           \
            LUISA_WARNING("Error occurred while parsing parameter, using default value: ", TO_STRING(default_value));  \
            return default_value;                                                                                      \
        }                                                                                                              \
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
    std::map<std::string, std::shared_ptr<Plugin>, std::less<>> _global_nodes;
    
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
            // FIXME: nullptr when using std::dynamic_pointer_cast
            return std::static_pointer_cast<T>(iter->second);
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
    return Plugin::create<BaseClass>(&_parser->device(), _derived_type_name, *this);
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

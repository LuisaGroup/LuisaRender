//
// Created by Mike on 2021/12/13.
//

#include <stdexcept>
#include <sdl/scene_node_desc.h>

namespace luisa::render {

void SceneNodeDesc::add_property(std::string_view name, SceneNodeDesc::value_list value) noexcept {
    if (!_properties.emplace(name, std::move(value)).second) {
        LUISA_ERROR_WITH_LOCATION(
            "Redefinition of property '{}' in "
            "scene description node '{}'.",
            name, _identifier);
    }
}

SceneNodeDesc *SceneNodeDesc::define_internal(std::string_view name, std::string_view impl_type, SourceLocation location) noexcept {
    auto unique_node = luisa::make_unique<SceneNodeDesc>(
        fmt::format("{}.$internal${}", _identifier, name),
        SceneNodeTag::INTERNAL);
    auto node = _internal_nodes.emplace_back(std::move(unique_node)).get();
    node->set_impl_type(impl_type);
    node->set_source_location(location);
    add_property(name, luisa::vector<node_type>{node});
    return node;
}

#define LUSIA_SCENE_NODE_DESC_PROPERTY_THROW(...) \
    throw std::runtime_error{fmt::format(__VA_ARGS__)};

#define LUISA_SCENE_NODE_DESC_PROPERTY_GET_POINTER(type)     \
    auto iter = _properties.find(name);                      \
    if (iter == _properties.cend()) [[unlikely]] {           \
        LUSIA_SCENE_NODE_DESC_PROPERTY_THROW(                \
            "Property '{}' is not defined in "               \
            "scene description node '{}'.",                  \
            name, _identifier);                              \
    }                                                        \
    auto ptr = [&] {                                         \
        if constexpr (std::is_same_v<type, int> ||           \
                      std::is_same_v<type, uint> ||          \
                      std::is_same_v<type, float>) {         \
            return std::get_if<number_list>(&iter->second);  \
        } else if constexpr (std::is_same_v<type, bool>) {   \
            return std::get_if<bool_list>(&iter->second);    \
        } else if constexpr (std::is_same_v<type, string> || \
                             std::is_same_v<type, path>) {   \
            return std::get_if<string_list>(&iter->second);  \
        } else {                                             \
            return std::get_if<node_list>(&iter->second);    \
        }                                                    \
    }();                                                     \
    if (ptr == nullptr) [[unlikely]] {                       \
        LUSIA_SCENE_NODE_DESC_PROPERTY_THROW(                \
            "Property '{}' is not a " #type " list in "      \
            "scene description node '{}'.",                  \
            name, _identifier);                              \
    }                                                        \
    auto size = ptr->size();                                 \
    auto convert = [&](size_t i) {                           \
        auto raw_value = (*ptr)[i];                          \
        if constexpr (std::is_same_v<type, int> ||           \
                      std::is_same_v<type, uint>) {          \
            auto value = static_cast<type>(raw_value);       \
            if (value != raw_value) [[unlikely]] {           \
                LUSIA_SCENE_NODE_DESC_PROPERTY_THROW(        \
                    "Invalid conversion from property '{}' " \
                    "(value = {}) to " #type                 \
                    "in scene description node '{}'.",       \
                    name, raw_value, _identifier);           \
            }                                                \
            return value;                                    \
        } else if constexpr (std::is_same_v<type, float> ||  \
                             std::is_same_v<type, path>) {   \
            return static_cast<type>(raw_value);             \
        } else {                                             \
            return raw_value;                                \
        }                                                    \
    };

#define LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR_OR_VECTOR(type, count)   \
    LUISA_SCENE_NODE_DESC_PROPERTY_GET_POINTER(type)                        \
    using namespace std::string_view_literals;                              \
    if (size < (count)) [[unlikely]] {                                      \
        LUSIA_SCENE_NODE_DESC_PROPERTY_THROW(                               \
            "Property '{}' in scene description node '{}' has {} value{}, " \
            "but is required to provide {} " #type " value{}.",             \
            name, _identifier, size, size <= 1u ? ""sv : "s"sv,             \
            count, (count) <= 1u ? ""sv : "s"sv);                           \
    }                                                                       \
    if (size > (count)) [[unlikely]] {                                      \
        LUISA_WARNING_WITH_LOCATION(                                        \
            "Property '{}' in scene description node '{}' "                 \
            "has {} values but is required to provide "                     \
            "only {} " #type " value{}. Remaining values "                  \
            "will be discarded.",                                           \
            name, _identifier, size,                                        \
            count, (count) <= 1u ? ""sv : "s"sv);                           \
    }

#define LUISA_SCENE_NODE_DESC_PROPERTY_HANDLE_DEFAULT(d, dv) \
    catch (const std::runtime_error &e) {                    \
        if constexpr (d) {                                   \
            return dv;                                       \
        } else {                                             \
            LUISA_ERROR_WITH_LOCATION("{}", e.what());       \
        }                                                    \
    }

#define LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_LIST(type, d, dv) \
    try {                                                     \
        LUISA_SCENE_NODE_DESC_PROPERTY_GET_POINTER(type)      \
        luisa::vector<type> values;                           \
        values.reserve(ptr->size());                          \
        for (auto i = 0u; i < size; i++) {                    \
            values.emplace_back(convert(i));                  \
        }                                                     \
        return values;                                        \
    }                                                         \
    LUISA_SCENE_NODE_DESC_PROPERTY_HANDLE_DEFAULT(d, dv)

#define LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR(type, d, dv)       \
    try {                                                             \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR_OR_VECTOR(type, 1) \
        return convert(0u);                                           \
    }                                                                 \
    LUISA_SCENE_NODE_DESC_PROPERTY_HANDLE_DEFAULT(d, dv)

#define LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_VECTOR2(type, d, dv)      \
    try {                                                             \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR_OR_VECTOR(type, 2) \
        return Vector<type, 2>{convert(0u), convert(1u)};             \
    }                                                                 \
    LUISA_SCENE_NODE_DESC_PROPERTY_HANDLE_DEFAULT(d, dv)

#define LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_VECTOR3(type, d, dv)      \
    try {                                                             \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR_OR_VECTOR(type, 3) \
        return Vector<type, 3>{                                       \
            convert(0u), convert(1u), convert(2u)};                   \
    }                                                                 \
    LUISA_SCENE_NODE_DESC_PROPERTY_HANDLE_DEFAULT(d, dv)

#define LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_VECTOR4(type, d, dv)      \
    try {                                                             \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR_OR_VECTOR(type, 4) \
        return Vector<type, 4>{                                       \
            convert(0u), convert(1u), convert(2u), convert(3u)};      \
    }                                                                 \
    LUISA_SCENE_NODE_DESC_PROPERTY_HANDLE_DEFAULT(d, dv)

#define LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL(type)                   \
    [[nodiscard]] type##_list SceneNodeDesc::property_##type##_list(            \
        std::string_view name) const noexcept {                                 \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_LIST(type, false, type##_list{})    \
    }                                                                           \
    [[nodiscard]] type##_list SceneNodeDesc::property_##type##_list_or_default( \
        std::string_view name, type##_list default_value) const noexcept {      \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_LIST(type, true, default_value)     \
    }

#define LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL(type)                               \
    [[nodiscard]] type SceneNodeDesc::property_##type(std::string_view name) const noexcept { \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR(type, false, type{})                       \
    }                                                                                         \
    [[nodiscard]] type SceneNodeDesc::property_##type##_or_default(                           \
        std::string_view name, type default_value) const noexcept {                           \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR(type, true, default_value)                 \
    }

#define LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(type, N)                                  \
    [[nodiscard]] type##N SceneNodeDesc::property_##type##N(std::string_view name) const noexcept { \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_VECTOR##N(type, false, type##N{})                       \
    }                                                                                               \
    [[nodiscard]] type##N SceneNodeDesc::property_##type##N##_or_default(                           \
        std::string_view name, type##N default_value) const noexcept {                              \
        LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_VECTOR##N(type, true, default_value)                    \
    }

using node = SceneNodeDesc::node;
using path = SceneNodeDesc::path;
using string = SceneNodeDesc::string;
using int_list = SceneNodeDesc::int_list;
using uint_list = SceneNodeDesc::uint_list;
using bool_list = SceneNodeDesc::bool_list;
using float_list = SceneNodeDesc::float_list;
using node_list = SceneNodeDesc::node_list;
using path_list = SceneNodeDesc::path_list;
using string_list = SceneNodeDesc::string_list;

LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL(int)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(int, 2)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(int, 3)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(int, 4)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL(uint)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(uint, 2)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(uint, 3)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(uint, 4)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL(bool)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(bool, 2)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(bool, 3)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(bool, 4)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL(float)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(float, 2)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(float, 3)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL(float, 4)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL(string)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL(path)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL(node)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL(int)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL(uint)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL(bool)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL(float)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL(string)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL(path)
LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL(node)

#undef LUSIA_SCENE_NODE_DESC_PROPERTY_THROW
#undef LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_LIST
#undef LUISA_SCENE_NODE_DESC_PROPERTY_GET_POINTER
#undef LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR
#undef LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_VECTOR2
#undef LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_VECTOR3
#undef LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_VECTOR4
#undef LUISA_SCENE_NODE_DESC_PROPERTY_HANDLE_DEFAULT
#undef LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_LIST_IMPL
#undef LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_SCALAR_IMPL
#undef LUISA_SCENE_NODE_DESC_PROPERTY_GETTER_VECTOR_IMPL
#undef LUISA_SCENE_NODE_DESC_PROPERTY_IMPL_SCALAR_OR_VECTOR

}// namespace luisa::render

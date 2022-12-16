//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <string>
#include <filesystem>

#include <core/hash.h>
#include <core/stl.h>
#include <core/logging.h>
#include <core/basic_types.h>
#include <sdl/scene_node_tag.h>

namespace luisa::render {

class SceneDesc;
class SceneNodeDesc;

namespace detail {

template<typename T>
struct scene_node_raw_property : public scene_node_raw_property<std::remove_cvref<T>> {};

template<>
struct scene_node_raw_property<bool> {
    using type = bool;
    static constexpr luisa::string_view value = "bool";
};

template<>
struct scene_node_raw_property<float> {
    using type = double;
    static constexpr luisa::string_view value = "number";
};

template<>
struct scene_node_raw_property<int> {
    using type = double;
    static constexpr luisa::string_view value = "number";
};

template<>
struct scene_node_raw_property<uint> {
    using type = double;
    static constexpr luisa::string_view value = "number";
};

template<>
struct scene_node_raw_property<luisa::string> {
    using type = luisa::string;
    static constexpr luisa::string_view value = "string";
};

template<>
struct scene_node_raw_property<std::filesystem::path> {
    using type = luisa::string;
    static constexpr luisa::string_view value = "path";
};

template<>
struct scene_node_raw_property<const SceneNodeDesc *> {
    using type = const SceneNodeDesc *;
    static constexpr luisa::string_view value = "node";
};

template<typename T>
using scene_node_raw_property_t = typename scene_node_raw_property<T>::type;

template<typename T>
constexpr auto scene_node_raw_property_v = scene_node_raw_property<T>::value;

template<typename T>
constexpr auto is_property_list_v = false;

template<typename T>
constexpr auto is_property_list_v<luisa::vector<T>> = true;

}// namespace detail

class SceneNodeDesc {

public:
    using bool_type = bool;
    using number_type = double;
    using string_type = luisa::string;
    using node_type = const SceneNodeDesc *;
    using bool_list = luisa::vector<bool_type>;
    using number_list = luisa::vector<number_type>;
    using string_list = luisa::vector<string_type>;
    using node_list = luisa::vector<node_type>;

    using value_list = luisa::variant<
        bool_list, number_list, string_list, node_list>;

    class SourceLocation {

    private:
        const std::filesystem::path *_file;
        uint32_t _line;
        uint32_t _column;

    public:
        SourceLocation() noexcept : _file{nullptr}, _line{}, _column{} {}
        explicit SourceLocation(const std::filesystem::path *path, uint32_t line = 0u, uint32_t col = 0u) noexcept
            : _file{path}, _line{line}, _column{col} {}
        [[nodiscard]] explicit operator bool() const noexcept { return _file != nullptr; }
        [[nodiscard]] auto file() const noexcept { return _file; }
        [[nodiscard]] auto line() const noexcept { return _line; }
        [[nodiscard]] auto column() const noexcept { return _column; }
        void set_line(uint32_t line) noexcept { _line = line; }
        void set_column(uint32_t col) noexcept { _column = col; }
        [[nodiscard]] auto string() const noexcept {
            using namespace std::string_literals;
            if (_file == nullptr) { return "unknown"s; }
            return fmt::format("{}:{}:{}", _file->string(), _line + 1u, _column);
        }
    };

private:
    luisa::string _identifier;
    luisa::string _impl_type;
    SourceLocation _location;
    const SceneNodeDesc *_base{nullptr};
    SceneNodeTag _tag;
    luisa::vector<luisa::unique_ptr<SceneNodeDesc>> _internal_nodes;
    luisa::unordered_map<luisa::string, value_list> _properties;

public:
    template<typename T>
    [[nodiscard]] luisa::optional<luisa::span<const detail::scene_node_raw_property_t<T>>>
    _property_raw_values(luisa::string_view name) const noexcept;
    template<typename T>
    [[nodiscard]] luisa::optional<T> _property_scalar(luisa::string_view name) const noexcept;
    template<typename T, size_t N>
    [[nodiscard]] luisa::optional<luisa::Vector<T, N>> _property_vector(luisa::string_view name) const noexcept;
    template<typename T>
    [[nodiscard]] luisa::optional<luisa::vector<T>> _property_list(luisa::string_view name) const noexcept;

    template<typename Dest, typename Src>
    [[nodiscard]] inline auto _property_convert(std::string_view name, Src &&src) const noexcept;

    template<typename T>
    [[nodiscard]] auto _property_generic(luisa::string_view name) const noexcept {
        if constexpr (is_vector_v<T>) {
            return _property_vector<vector_element_t<T>, vector_dimension_v<T>>(name);
        } else if constexpr (detail::is_property_list_v<T>) {
            return _property_list<typename T::value_type>(name);
        } else {
            return _property_scalar<T>(name);
        }
    }

public:
    SceneNodeDesc(luisa::string identifier, SceneNodeTag tag) noexcept
        : _identifier{std::move(identifier)}, _tag{tag} {}
    SceneNodeDesc(SceneNodeDesc &&) noexcept = delete;
    SceneNodeDesc(const SceneNodeDesc &) noexcept = delete;
    SceneNodeDesc &operator=(SceneNodeDesc &&) noexcept = delete;
    SceneNodeDesc &operator=(const SceneNodeDesc &) noexcept = delete;
    [[nodiscard]] auto identifier() const noexcept { return luisa::string_view{_identifier}; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto impl_type() const noexcept { return luisa::string_view{_impl_type}; }
    [[nodiscard]] auto source_location() const noexcept { return _location; }
    void define(SceneNodeTag tag, luisa::string_view t, SourceLocation l, const SceneNodeDesc *base = nullptr) noexcept;
    [[nodiscard]] auto &properties() const noexcept { return _properties; }
    [[nodiscard]] bool has_property(luisa::string_view prop) const noexcept;
    void add_property(luisa::string_view name, value_list values) noexcept;
    void add_property(luisa::string_view name, bool_type value) noexcept { add_property(name, bool_list{value}); }
    void add_property(luisa::string_view name, number_type value) noexcept { add_property(name, number_list{value}); }
    void add_property(luisa::string_view name, string_type value) noexcept { add_property(name, string_list{std::move(value)}); }
    void add_property(luisa::string_view name, const char *value) noexcept { add_property(name, string_list{value}); }
    void add_property(luisa::string_view name, node_type value) noexcept { add_property(name, node_list{value}); }
    [[nodiscard]] SceneNodeDesc *define_internal(
        luisa::string_view impl_type, SourceLocation location = {}, const SceneNodeDesc *base = nullptr) noexcept;
    [[nodiscard]] auto is_root() const noexcept { return _tag == SceneNodeTag::ROOT; }
    [[nodiscard]] auto is_internal() const noexcept { return _tag == SceneNodeTag::INTERNAL; }
    [[nodiscard]] auto is_defined() const noexcept { return _tag != SceneNodeTag::DECLARATION && !_impl_type.empty(); }
    [[nodiscard]] static const SceneNodeDesc *shared_default(SceneNodeTag tag, luisa::string impl) noexcept;
#define LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(category, tag)                            \
    [[nodiscard]] static auto shared_default_##category(luisa::string impl) noexcept { \
        return shared_default(SceneNodeTag::tag, std::move(impl));                     \
    }
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(camera, CAMERA)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(shape, SHAPE)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(surface, SURFACE)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(light, LIGHT)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(transform, TRANSFORM)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(film, FILM)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(filter, FILTER)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(sampler, SAMPLER)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(integrator, INTEGRATOR)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(light_sampler, LIGHT_SAMPLER)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(environment, ENVIRONMENT)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(texture, TEXTURE)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(texture_mapping, TEXTURE_MAPPING)
    LUISA_SCENE_NODE_DESC_SHARED_DEFAULT(spectrum, SPECTRUM)
#undef LUISA_SCENE_NODE_DESC_SHARED_DEFAULT

public:
    using int_list = luisa::vector<int>;
    using uint_list = luisa::vector<uint>;
    using float_list = luisa::vector<float>;
    using node = const SceneNodeDesc *;
    using string = luisa::string;
    using path = std::filesystem::path;
    using path_list = luisa::vector<path>;

    // parameter getters
#define LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(type)                      \
    [[nodiscard]] type property_##type(                                  \
        std::string_view name) const noexcept {                          \
        if (auto prop = _property_generic<type>(name)) [[likely]] {      \
            return *prop;                                                \
        }                                                                \
        LUISA_ERROR(                                                     \
            "No valid values given for property '{}' in "                \
            "scene description node '{}'. [{}]",                         \
            name, _identifier, source_location().string());              \
    }                                                                    \
    template<typename DV = type>                                         \
    [[nodiscard]] type property_##type##_or_default(                     \
        std::string_view name,                                           \
        DV &&default_value = std::remove_cvref_t<DV>{}) const noexcept { \
        return _property_generic<type>(name).value_or(                   \
            std::forward<DV>(default_value));                            \
    }
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(int)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(int2)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(int3)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(int4)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(uint)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(uint2)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(uint3)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(uint4)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(bool)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(bool2)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(bool3)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(bool4)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(float)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(float2)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(float3)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(float4)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(string)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(path)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(node)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(int_list)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(uint_list)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(bool_list)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(float_list)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(string_list)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(path_list)
    LUISA_SCENE_NODE_DESC_PROPERTY_GETTER(node_list)
#undef LUISA_SCENE_NODE_DESC_PROPERTY_GETTER
};

template<typename Dest, typename Src>
auto SceneNodeDesc::_property_convert(std::string_view name, Src &&src) const noexcept {
    auto sloc = source_location();
    if constexpr (std::is_same_v<Dest, path>) {
        SceneNodeDesc::path p{std::forward<Src>(src)};
        if (!sloc || p.is_absolute()) { return p; }
        return std::filesystem::canonical(sloc.file()->parent_path()) / p;
    } else if constexpr (std::is_same_v<Dest, int> || std::is_same_v<Dest, uint>) {
        auto value = static_cast<Dest>(src);
        if (value != src) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Cannot property '{}' (value = {}) to integer "
                "in scene description node '{}'. [{}]",
                name, src, _identifier, sloc.string());
        }
        return value;
    } else {
        return static_cast<Dest>(std::forward<Src>(src));
    }
}

template<typename T>
inline luisa::optional<luisa::span<const detail::scene_node_raw_property_t<T>>>
SceneNodeDesc::_property_raw_values(luisa::string_view name) const noexcept {
    auto iter = _properties.find(name);
    if (iter == _properties.cend()) {
        return _base == nullptr ?
                   luisa::nullopt :
                   _base->_property_raw_values<T>(name);
    }
    using raw_type = detail::scene_node_raw_property_t<T>;
    auto ptr = luisa::get_if<luisa::vector<raw_type>>(&iter->second);
    if (ptr == nullptr) [[unlikely]] {
        LUISA_WARNING(
            "Property '{}' is defined but is not a {} list "
            "in scene description node '{}'. [{}]",
            name, detail::scene_node_raw_property_v<T>,
            _identifier, source_location().string());
        return luisa::nullopt;
    }
    return luisa::span{std::as_const(*ptr)};
}

template<typename T>
inline optional<T> SceneNodeDesc::_property_scalar(luisa::string_view name) const noexcept {
    if (auto raw_values_opt = _property_raw_values<T>(name);
        raw_values_opt && !raw_values_opt->empty()) [[likely]] {
        auto raw_values = *raw_values_opt;
        if (raw_values.size() > 1u) [[unlikely]] {
            LUISA_WARNING(
                "Found {} values given for property '{}' in "
                "scene description node '{}', but only 1 is required. "
                "Additional values will be discarded. [{}]",
                raw_values.size(), name, _identifier, source_location().string());
        }
        return _property_convert<T>(name, raw_values.front());
    }
    return luisa::nullopt;
}

template<typename T, size_t N>
inline optional<luisa::Vector<T, N>> SceneNodeDesc::_property_vector(luisa::string_view name) const noexcept {
    if (auto raw_values_opt = _property_raw_values<T>(name);
        raw_values_opt && !raw_values_opt->empty()) [[likely]] {
        auto raw_values = *raw_values_opt;
        if (raw_values.size() < N) [[unlikely]] {
            LUISA_WARNING(
                "Required {} values but found {} for property '{}' "
                "in scene description node '{}'. [{}]",
                N, raw_values.size(), name, _identifier,
                source_location().string());
            return luisa::nullopt;
        }
        if (raw_values.size() > N) [[unlikely]] {
            LUISA_WARNING(
                "Required {} values but found {} for property '{}' "
                "in scene description node '{}'. "
                "Additional values will be discarded. [{}]",
                N, raw_values.size(), name, _identifier,
                source_location().string());
        }
        luisa::Vector<T, N> v;
        for (auto i = 0u; i < N; i++) {
            v[i] = _property_convert<T>(name, raw_values[i]);
        }
        return v;
    }
    return luisa::nullopt;
}

template<typename T>
inline luisa::optional<luisa::vector<T>> SceneNodeDesc::_property_list(luisa::string_view name) const noexcept {
    if (auto raw_values_opt = _property_raw_values<T>(name)) [[likely]] {
        auto raw_values = *raw_values_opt;
        luisa::vector<T> values;
        values.reserve(raw_values.size());
        for (auto &&v : raw_values) {
            values.emplace_back(_property_convert<T>(name, v));
        }
        return values;
    }
    return luisa::nullopt;
}

}// namespace luisa::render

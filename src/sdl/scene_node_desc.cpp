//
// Created by Mike on 2021/12/13.
//

#include <stdexcept>

#include <core/logging.h>
#include <sdl/scene_node_desc.h>

namespace luisa::render {

void SceneNodeDesc::add_property(luisa::string_view name, SceneNodeDesc::value_list value) noexcept {
    if (!_properties.emplace(luisa::string{name}, std::move(value)).second) {
        LUISA_ERROR(
            "Redefinition of property '{}' in "
            "scene description node '{}'. [{}]",
            name, _identifier, _location.string());
    }
}

void SceneNodeDesc::define(SceneNodeTag tag, luisa::string_view t, SceneNodeDesc::SourceLocation l, const SceneNodeDesc *base) noexcept {
    _tag = tag;
    _location = l;
    _impl_type = t;
    _base = base;
    for (auto &c : _impl_type) {
        c = static_cast<char>(tolower(c));
    }
}

SceneNodeDesc *SceneNodeDesc::define_internal(
    luisa::string_view impl_type, SourceLocation location, const SceneNodeDesc *base) noexcept {
    auto unique_node = luisa::make_unique<SceneNodeDesc>(
        fmt::format("{}.$internal{}", _identifier, _internal_nodes.size()),
        SceneNodeTag::INTERNAL);
    auto node = _internal_nodes.emplace_back(std::move(unique_node)).get();
    node->define(SceneNodeTag::INTERNAL, impl_type, location, base);
    return node;
}

bool SceneNodeDesc::has_property(luisa::string_view prop) const noexcept {
    return _properties.find_as(prop, Hash64{}, std::equal_to<>{}) != _properties.cend() ||
           (_base != nullptr && _base->has_property(prop));
}

}// namespace luisa::render

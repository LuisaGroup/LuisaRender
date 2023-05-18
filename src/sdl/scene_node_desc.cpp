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

void SceneNodeDesc::define(SceneNodeTag tag, luisa::string_view t,
                           SceneNodeDesc::SourceLocation l, const SceneNodeDesc *base) noexcept {
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
        luisa::format("{}.$internal{}", _identifier, _internal_nodes.size()),
        SceneNodeTag::INTERNAL);
    auto node = _internal_nodes.emplace_back(std::move(unique_node)).get();
    node->define(SceneNodeTag::INTERNAL, impl_type, location, base);
    return node;
}

bool SceneNodeDesc::has_property(luisa::string_view prop) const noexcept {
    return _properties.find(prop) != _properties.cend() ||
           (_base != nullptr && _base->has_property(prop));
}

const SceneNodeDesc *SceneNodeDesc::shared_default(SceneNodeTag tag, luisa::string impl) noexcept {
    static luisa::unordered_map<uint64_t, luisa::unique_ptr<SceneNodeDesc>> descriptions;
    static std::mutex mutex;
    static thread_local const auto seed = hash_value("__scene_node_tag_and_impl_type_hash");
    for (auto &c : impl) { c = static_cast<char>(tolower(c)); }
    auto hash = hash_value(impl, hash_value(to_underlying(tag), seed));
    std::scoped_lock lock{mutex};
    if (auto iter = descriptions.find(hash);
        iter != descriptions.cend()) { return iter->second.get(); }
    auto identifier = luisa::format(
        "__shared_default_{}_{}",
        scene_node_tag_description(tag), impl);
    for (auto &c : identifier) { c = static_cast<char>(tolower(c)); }
    auto desc = luisa::make_unique<SceneNodeDesc>(
        std::move(identifier), tag);
    desc->define(tag, impl, {});
    return descriptions.emplace(hash, std::move(desc)).first->second.get();
}

}// namespace luisa::render

//
// Created by Mike on 2021/12/8.
//

#include <mutex>
#include <fmt/format.h>
#include <base/scene.h>

namespace luisa::render {

namespace detail {

[[nodiscard]] static auto &scene_plugin_registry() noexcept {
    static luisa::unordered_map<luisa::string, luisa::unique_ptr<DynamicModule>> registry;
    return registry;
}

[[nodiscard]] static auto &scene_plugin_registry_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] static auto &scene_plugin_load(const std::filesystem::path &runtime_dir, SceneNode::Tag tag, std::string_view impl_type) noexcept {
    std::scoped_lock lock{detail::scene_plugin_registry_mutex()};
    luisa::string name{fmt::format("luisa-render-{}-{}", SceneNode::tag_description(tag), impl_type)};
    for (auto &c : name) { c = static_cast<char>(std::tolower(c)); }
    auto &&registry = detail::scene_plugin_registry();
    if (auto iter = registry.find(name); iter != registry.end()) {
        return *iter->second;
    }
    auto module = luisa::make_unique<DynamicModule>(runtime_dir, name);
    return *registry.emplace(name, std::move(module)).first->second;
}

}// namespace detail

SceneNode *Scene::add(SceneNode::Tag tag, std::string_view identifier, std::string_view impl_type) noexcept {
    if (auto iter = _nodes.find(identifier); iter != _nodes.end()) {
        LUISA_ERROR_WITH_LOCATION(
            "Scene node `{}` (type = {}::{}) is already in the graph (type = {}::{}).",
            identifier, SceneNode::tag_description(tag), impl_type,
            SceneNode::tag_description(iter->second->tag()),
            iter->second->impl_type());
    }
    auto &&plugin = detail::scene_plugin_load(_context->runtime_directory() / "plugins", tag, impl_type);
    auto node = plugin.invoke<Node *(void)>("create");
    auto deleter = plugin.function<void(Node *)>("destroy");
    auto ptr = _nodes.emplace(identifier, std::unique_ptr<Node, void (*)(Node *)>{node, deleter}).first->second.get();
    if (ptr->tag() != tag || ptr->impl_type() != impl_type) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Scene node `{}` (type = {}::{}) is created with invalid type {}::{}.",
            identifier, SceneNode::tag_description(tag), impl_type,
            SceneNode::tag_description(ptr->tag()), ptr->impl_type());
    }
    return ptr;
}

Scene::Scene(const Context &ctx) noexcept : _context{&ctx} {}
Scene::~Scene() noexcept = default;

}// namespace luisa::render

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

[[nodiscard]] constexpr auto scene_node_tag_description(Scene::Node::Tag tag) noexcept -> std::string_view {
    using namespace std::string_view_literals;
    switch (tag) {
        case Scene::Node::Tag::CAMERA: return "camera"sv;
        case Scene::Node::Tag::SHAPE: return "shape"sv;
        case Scene::Node::Tag::MATERIAL: return "material"sv;
        case Scene::Node::Tag::TRANSFORM: return "transform"sv;
        case Scene::Node::Tag::FILM: return "film"sv;
        case Scene::Node::Tag::FILTER: return "filter"sv;
        case Scene::Node::Tag::SAMPLER: return "sampler"sv;
        case Scene::Node::Tag::INTEGRATOR: return "integrator"sv;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Known scene node tag: 0x{:x}.",
        to_underlying(tag));
}

}// namespace detail

DynamicModule &Scene::_load_plugin(Scene::Node::Tag tag, std::string_view impl_type) noexcept {
    std::scoped_lock lock{detail::scene_plugin_registry_mutex()};
    luisa::string name{fmt::format("luisa-render-{}-{}", detail::scene_node_tag_description(tag), impl_type)};
    auto &&registry = detail::scene_plugin_registry();
    if (auto iter = registry.find(name); iter != registry.end()) {
        return *iter->second;
    }
    auto module = luisa::make_unique<DynamicModule>(_context->runtime_directory() / "plugins", name);
    return *registry.emplace(name, std::move(module)).first->second;
}

Scene::Node *Scene::add(Scene::Node::Tag tag, std::string_view identifier, std::string_view impl_type) noexcept {
    if (auto iter = _nodes.find(identifier); iter != _nodes.end()) {
        LUISA_ERROR_WITH_LOCATION(
            "Scene node `{}` (type = {}::{}) is already in the graph (type = {}).",
            identifier, detail::scene_node_tag_description(tag), impl_type,
            detail::scene_node_tag_description(iter->second->tag()));
    }
    auto &&plugin = _load_plugin(tag, impl_type);
    auto node = plugin.invoke<Node *(void)>("create");
    auto deleter = plugin.function<void(Node *)>("destroy");
    auto ptr = _nodes.emplace(identifier, std::unique_ptr<Node, void (*)(Node *)>{node, deleter}).first->second.get();
    if (auto created_tag = ptr->tag(); created_tag != tag) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Scene node `{}` (type = {}::{}) is created with invalid type {}.",
            identifier, detail::scene_node_tag_description(tag), impl_type,
            detail::scene_node_tag_description(created_tag));
    }
    return ptr;
}

Scene::Scene(const Context &ctx) noexcept : _context{&ctx} {}

Scene::~Scene() noexcept = default;

}// namespace luisa::render

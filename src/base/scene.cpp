//
// Created by Mike on 2021/12/8.
//

#include <mutex>
#include <fmt/format.h>

#include <base/camera.h>
#include <base/film.h>
#include <base/filter.h>
#include <base/integrator.h>
#include <base/material.h>
#include <base/sampler.h>
#include <base/shape.h>
#include <base/transform.h>
#include <base/environment.h>
#include <base/scene_desc.h>
#include <base/scene_node_desc.h>
#include <base/scene.h>

namespace luisa::render {

struct Scene::Config {
    luisa::vector<NodeHandle> internal_nodes;
    luisa::unordered_map<luisa::string, NodeHandle, Hash64> nodes;
    Integrator *integrator{nullptr};
    luisa::vector<Camera *> cameras;
    luisa::vector<Shape *> shapes;
    luisa::vector<Environment *> environments;
};

const Integrator *Scene::integrator() const noexcept { return _config->integrator; }
std::span<const Shape *const> Scene::shapes() const noexcept { return _config->shapes; }
std::span<const Camera *const> Scene::cameras() const noexcept { return _config->cameras; }
std::span<const Environment *const> Scene::environments() const noexcept { return _config->environments; }

namespace detail {

[[nodiscard]] static auto &scene_plugin_registry() noexcept {
    static luisa::unordered_map<luisa::string, luisa::unique_ptr<DynamicModule>, Hash64> registry;
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

SceneNode *Scene::_node(SceneNode::Tag tag, const SceneNodeDesc *desc) noexcept {
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description "
            "node '{}' (type = {}::{}).",
            desc->identifier(),
            SceneNode::tag_description(desc->tag()),
            desc->impl_type());
    }
    auto &&plugin = detail::scene_plugin_load(
        _context.runtime_directory() / "plugins",
        tag, desc->impl_type());
    auto create = plugin.function<NodeCreater>("create");
    auto destroy = plugin.function<NodeDeleter>("destroy");
    if (desc->is_internal()) {
        NodeHandle node{create(this, desc), destroy};
        return _config->internal_nodes.emplace_back(std::move(node)).get();
    }
    if (desc->tag() != tag) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid tag {} of scene description "
            "node '{}' (expected {}).",
            SceneNode::tag_description(desc->tag()),
            desc->identifier(),
            SceneNode::tag_description(tag));
    }
    auto [iter, first_def] = _config->nodes.try_emplace(
        luisa::string{desc->identifier()},
        lazy_construct([desc, create, destroy, this] {
            return NodeHandle{create(this, desc), destroy};
        }));
    auto node = iter->second.get();
    if (first_def) { return node; }
    if (node->tag() != tag || node->impl_type() != desc->impl_type()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Scene node `{}` (type = {}::{}) is already in the graph (type = {}::{}).",
            desc->identifier(), SceneNode::tag_description(tag),
            desc->impl_type(), SceneNode::tag_description(node->tag()),
            node->impl_type());
    }
    return node;
}

inline Scene::Scene(const Context &ctx) noexcept
    : _context{ctx},
      _config{luisa::make_unique<Scene::Config>()} {}

Camera *Scene::_camera(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Camera *>(_node(SceneNode::Tag::CAMERA, desc));
}

Film *Scene::_film(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Film *>(_node(SceneNode::Tag::FILM, desc));
}

Filter *Scene::_filter(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Filter *>(_node(SceneNode::Tag::FILTER, desc));
}

Integrator *Scene::_integrator(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Integrator *>(_node(SceneNode::Tag::INTEGRATOR, desc));
}

Material *Scene::_material(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Material *>(_node(SceneNode::Tag::MATERIAL, desc));
}

Sampler *Scene::_sampler(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Sampler *>(_node(SceneNode::Tag::SAMPLER, desc));
}

Shape *Scene::_shape(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Shape *>(_node(SceneNode::Tag::SHAPE, desc));
}

Transform *Scene::_transform(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Transform *>(_node(SceneNode::Tag::TRANSFORM, desc));
}

Environment *Scene::_environment(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Environment *>(_node(SceneNode::Tag::ENVIRONMENT, desc));
}

luisa::unique_ptr<Scene> Scene::create(const Context &ctx, const SceneDesc *desc) noexcept {
    if (!desc->root()->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Root node is not defined "
            "in the scene description.");
    }
    auto scene = luisa::make_unique<Scene>(ctx);
    scene->_config->integrator = scene->_integrator(desc->root()->property_node("integrator"));
    auto cameras = desc->root()->property_node_list("cameras");
    auto shapes = desc->root()->property_node_list("shapes");
    auto environments = desc->root()->property_node_list("environments");
    scene->_config->cameras.reserve(cameras.size());
    scene->_config->shapes.reserve(shapes.size());
    scene->_config->environments.reserve(environments.size());
    for (auto c : cameras) {
        scene->_config->cameras.emplace_back(
            scene->_camera(c));
    }
    for (auto s : shapes) {
        scene->_config->shapes.emplace_back(
            scene->_shape(s));
    }
    for (auto e : environments) {
        scene->_config->environments.emplace_back(
            scene->_environment(e));
    }
    return scene;
}

Scene::~Scene() noexcept = default;

}// namespace luisa::render

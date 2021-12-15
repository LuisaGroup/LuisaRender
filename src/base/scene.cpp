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
#include <base/light.h>
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
    uint2 resolution{};
    uint spp{0u};
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

SceneNode *Scene::load_node(SceneNode::Tag tag, const SceneNodeDesc *desc) noexcept {
    if (desc == nullptr) { return nullptr; }
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

Camera *Scene::load_camera(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Camera *>(load_node(SceneNode::Tag::CAMERA, desc));
}

Film *Scene::load_film(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Film *>(load_node(SceneNode::Tag::FILM, desc));
}

Filter *Scene::load_filter(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Filter *>(load_node(SceneNode::Tag::FILTER, desc));
}

Integrator *Scene::load_integrator(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Integrator *>(load_node(SceneNode::Tag::INTEGRATOR, desc));
}

Material *Scene::load_material(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Material *>(load_node(SceneNode::Tag::MATERIAL, desc));
}

Light *Scene::load_light(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Light *>(load_node(SceneNode::Tag::LIGHT, desc));
}

Sampler *Scene::load_sampler(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Sampler *>(load_node(SceneNode::Tag::SAMPLER, desc));
}

Shape *Scene::load_shape(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Shape *>(load_node(SceneNode::Tag::SHAPE, desc));
}

Transform *Scene::load_transform(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Transform *>(load_node(SceneNode::Tag::TRANSFORM, desc));
}

Environment *Scene::load_environment(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Environment *>(load_node(SceneNode::Tag::ENVIRONMENT, desc));
}

luisa::unique_ptr<Scene> Scene::create(const Context &ctx, const SceneDesc *desc) noexcept {
    if (!desc->root()->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Root node is not defined "
            "in the scene description.");
    }
    auto scene = luisa::make_unique<Scene>(ctx);
    scene->_config->resolution = desc->root()->property_uint2_or_default(
        "resolution",
        make_uint2(desc->root()->property_uint_or_default(
            "resolution", 1024u)));
    scene->_config->spp = desc->root()->property_uint_or_default("spp", 1024u);
    scene->_config->integrator = scene->load_integrator(desc->root()->property_node("integrator"));
    auto cameras = desc->root()->property_node_list("cameras");
    auto shapes = desc->root()->property_node_list("shapes");
    auto environments = desc->root()->property_node_list("environments");
    scene->_config->cameras.reserve(cameras.size());
    scene->_config->shapes.reserve(shapes.size());
    scene->_config->environments.reserve(environments.size());
    for (auto c : cameras) {
        scene->_config->cameras.emplace_back(
            scene->load_camera(c));
    }
    for (auto s : shapes) {
        scene->_config->shapes.emplace_back(
            scene->load_shape(s));
    }
    for (auto e : environments) {
        scene->_config->environments.emplace_back(
            scene->load_environment(e));
    }
    return scene;
}

uint2 Scene::resolution() const noexcept { return _config->resolution; }
uint Scene::spp() const noexcept { return _config->spp; }

Scene::~Scene() noexcept = default;

}// namespace luisa::render

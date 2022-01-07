//
// Created by Mike on 2021/12/8.
//

#include <mutex>
#include <fmt/format.h>

#include <scene/camera.h>
#include <scene/film.h>
#include <scene/filter.h>
#include <scene/integrator.h>
#include <scene/material.h>
#include <scene/light.h>
#include <scene/sampler.h>
#include <scene/shape.h>
#include <scene/transform.h>
#include <scene/environment.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_node_desc.h>
#include <scene/scene.h>

namespace luisa::render {

struct Scene::Config {
    luisa::vector<NodeHandle> internal_nodes;
    luisa::unordered_map<luisa::string, NodeHandle, Hash64> nodes;
    Integrator *integrator{nullptr};
    luisa::vector<Camera *> cameras;
    luisa::vector<Shape *> shapes;
    luisa::vector<Environment *> environments;
    uint spp{0u};
};

uint Scene::spp() const noexcept { return _config->spp; }
const Integrator *Scene::integrator() const noexcept { return _config->integrator; }
luisa::span<const Shape *const> Scene::shapes() const noexcept { return _config->shapes; }
luisa::span<const Camera *const> Scene::cameras() const noexcept { return _config->cameras; }
luisa::span<const Environment *const> Scene::environments() const noexcept { return _config->environments; }

namespace detail {

[[nodiscard]] static auto &scene_plugin_registry() noexcept {
    static luisa::unordered_map<luisa::string, luisa::unique_ptr<DynamicModule>, Hash64> registry;
    return registry;
}

[[nodiscard]] static auto &scene_plugin_registry_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] static auto &scene_plugin_load(const std::filesystem::path &runtime_dir, SceneNodeTag tag, std::string_view impl_type) noexcept {
    std::scoped_lock lock{detail::scene_plugin_registry_mutex()};
    luisa::string name{fmt::format("luisa-render-{}-{}", scene_node_tag_description(tag), impl_type)};
    for (auto &c : name) { c = static_cast<char>(std::tolower(c)); }
    auto &&registry = detail::scene_plugin_registry();
    if (auto iter = registry.find(name); iter != registry.end()) {
        return *iter->second;
    }
    auto module = luisa::make_unique<DynamicModule>(runtime_dir, name);
    return *registry.emplace(name, std::move(module)).first->second;
}

}// namespace detail

SceneNode *Scene::load_node(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept {
    if (desc == nullptr) { return nullptr; }
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description "
            "node '{}' (type = {}::{}).",
            desc->identifier(),
            scene_node_tag_description(desc->tag()),
            desc->impl_type());
    }
    auto &&plugin = detail::scene_plugin_load(
        _context.runtime_directory() / "plugins",
        tag, desc->impl_type());
    auto create = plugin.function<NodeCreater>("create");
    auto destroy = plugin.function<NodeDeleter>("destroy");
    if (desc->is_internal()) {
        NodeHandle node{create(this, desc), destroy};
        std::scoped_lock lock{_mutex};
        return _config->internal_nodes.emplace_back(std::move(node)).get();
    }
    if (desc->tag() != tag) [[unlikely]] {
        LUISA_ERROR(
            "Invalid tag {} of scene description "
            "node '{}' (expected {}). [{}]",
            scene_node_tag_description(desc->tag()),
            desc->identifier(),
            scene_node_tag_description(tag),
            desc->source_location().string());
    }

    auto [node, first_def] = [this, desc, create, destroy] {
        std::scoped_lock lock{_mutex};
        auto [iter, first] = _config->nodes.try_emplace(
            luisa::string{desc->identifier()},
            lazy_construct([desc, create, destroy, this] {
                return NodeHandle{create(this, desc), destroy};
            }));
        return std::make_pair(iter->second.get(), first);
    }();
    if (first_def) { return node; }
    if (node->tag() != tag || node->impl_type() != desc->impl_type()) [[unlikely]] {
        LUISA_ERROR(
            "Scene node `{}` (type = {}::{}) is already "
            "in the graph (type = {}::{}). [{}]",
            desc->identifier(), scene_node_tag_description(tag),
            desc->impl_type(), scene_node_tag_description(node->tag()),
            node->impl_type(), desc->source_location().string());
    }
    return node;
}

inline Scene::Scene(const Context &ctx) noexcept
    : _context{ctx},
      _config{luisa::make_unique<Scene::Config>()} {}

Camera *Scene::load_camera(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Camera *>(load_node(SceneNodeTag::CAMERA, desc));
}

Film *Scene::load_film(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Film *>(load_node(SceneNodeTag::FILM, desc));
}

Filter *Scene::load_filter(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Filter *>(load_node(SceneNodeTag::FILTER, desc));
}

Integrator *Scene::load_integrator(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Integrator *>(load_node(SceneNodeTag::INTEGRATOR, desc));
}

Material *Scene::load_material(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Material *>(load_node(SceneNodeTag::MATERIAL, desc));
}

Light *Scene::load_light(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Light *>(load_node(SceneNodeTag::LIGHT, desc));
}

Sampler *Scene::load_sampler(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Sampler *>(load_node(SceneNodeTag::SAMPLER, desc));
}

Shape *Scene::load_shape(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Shape *>(load_node(SceneNodeTag::SHAPE, desc));
}

Transform *Scene::load_transform(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Transform *>(load_node(SceneNodeTag::TRANSFORM, desc));
}

Environment *Scene::load_environment(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Environment *>(load_node(SceneNodeTag::ENVIRONMENT, desc));
}

luisa::unique_ptr<Scene> Scene::create(const Context &ctx, const SceneDesc *desc) noexcept {
    if (!desc->root()->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Root node is not defined "
            "in the scene description.");
    }
    auto scene = luisa::make_unique<Scene>(ctx);
    scene->_config->spp = desc->root()->property_uint_or_default("spp", 1024u);
    scene->_config->integrator = scene->load_integrator(desc->root()->property_node("integrator"));
    auto cameras = desc->root()->property_node_list("cameras");
    auto shapes = desc->root()->property_node_list("shapes");
    auto environments = desc->root()->property_node_list_or_default("environments");
    scene->_config->cameras.reserve(cameras.size());
    scene->_config->shapes.reserve(shapes.size());
    scene->_config->environments.reserve(environments.size());
    // TODO: parallel loading
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

Scene::~Scene() noexcept = default;

}// namespace luisa::render

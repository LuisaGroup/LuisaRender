//
// Created by Mike on 2021/12/8.
//

#include <mutex>

#include <core/thread_pool.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_node_desc.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/filter.h>
#include <base/integrator.h>
#include <base/surface.h>
#include <base/light.h>
#include <base/sampler.h>
#include <base/shape.h>
#include <base/transform.h>
#include <base/environment.h>
#include <base/light_sampler.h>
#include <base/texture.h>
#include <base/texture_mapping.h>
#include <base/spectrum.h>
#include <base/scene.h>

namespace luisa::render {

struct Scene::Config {
    luisa::vector<NodeHandle> internal_nodes;
    luisa::unordered_map<luisa::string, NodeHandle, Hash64, std::equal_to<>> nodes;
    Integrator *integrator{nullptr};
    Environment *environment{nullptr};
    luisa::vector<Camera *> cameras;
    luisa::vector<Shape *> shapes;
};

const Integrator *Scene::integrator() const noexcept { return _config->integrator; }
const Environment *Scene::environment() const noexcept { return _config->environment; }
luisa::span<const Shape *const> Scene::shapes() const noexcept { return _config->shapes; }
luisa::span<const Camera *const> Scene::cameras() const noexcept { return _config->cameras; }

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
        _context.runtime_directory(), tag, desc->impl_type());
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
    auto [node, first_def] = [&] {
        std::scoped_lock lock{_mutex};
        if (auto iter = _config->nodes.find(desc->identifier());
            iter != _config->nodes.end()) {
            return std::make_pair(iter->second.get(), false);
        }
        LUISA_VERBOSE_WITH_LOCATION(
            "Constructing scene graph node '{}' (desc = {}).",
            desc->identifier(), fmt::ptr(desc));
        NodeHandle new_node{create(this, desc), destroy};
        auto ptr = new_node.get();
        _config->nodes.emplace(desc->identifier(), std::move(new_node));
        return std::make_pair(ptr, true);
    }();
    if (!first_def && (node->tag() != tag ||
                       node->impl_type() != desc->impl_type())) [[unlikely]] {
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

Surface *Scene::load_surface(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Surface *>(load_node(SceneNodeTag::SURFACE, desc));
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

LightSampler *Scene::load_light_sampler(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<LightSampler *>(load_node(SceneNodeTag::LIGHT_SAMPLER, desc));
}

Environment *Scene::load_environment(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Environment *>(load_node(SceneNodeTag::ENVIRONMENT, desc));
}

Texture *Scene::load_texture(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Texture *>(load_node(SceneNodeTag::TEXTURE, desc));
}

TextureMapping *Scene::load_texture_mapping(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<TextureMapping *>(load_node(SceneNodeTag::TEXTURE_MAPPING, desc));
}

Spectrum *Scene::load_spectrum(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Spectrum *>(load_node(SceneNodeTag::SPECTRUM, desc));
}

Loss *Scene::load_loss(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Loss *>(load_node(SceneNodeTag::LOSS, desc));
}

Optimizer *Scene::load_optimizer(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Optimizer *>(load_node(SceneNodeTag::OPTIMIZER, desc));
}

luisa::unique_ptr<Scene> Scene::create(const Context &ctx, const SceneDesc *desc) noexcept {
    if (!desc->root()->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Root node is not defined "
            "in the scene description.");
    }
    auto scene = luisa::make_unique<Scene>(ctx);
    scene->_config->integrator = scene->load_integrator(desc->root()->property_node("integrator"));
    scene->_config->environment = scene->load_environment(desc->root()->property_node_or_default("environment"));
    auto cameras = desc->root()->property_node_list("cameras");
    auto shapes = desc->root()->property_node_list("shapes");
    auto environments = desc->root()->property_node_list_or_default("environments");
    scene->_config->cameras.reserve(cameras.size());
    scene->_config->shapes.reserve(shapes.size());
    for (auto c : cameras) {
        scene->_config->cameras.emplace_back(
            scene->load_camera(c));
    }
    for (auto s : shapes) {
        scene->_config->shapes.emplace_back(
            scene->load_shape(s));
    }
    ThreadPool::global().synchronize();
    if (!scene->_config->integrator->is_differentiable()) {
        auto disabled = 0u;
        for (auto &&node : scene->_config->internal_nodes) {
            if (node->tag() == SceneNodeTag::TEXTURE) {
                auto texture = static_cast<Texture *>(node.get());
                if (texture->requires_gradients()) {
                    disabled++;
                    texture->disable_gradients();
                }
            }
        }
        for (auto &&[_, node] : scene->_config->nodes) {
            if (node->tag() == SceneNodeTag::TEXTURE) {
                auto texture = static_cast<Texture *>(node.get());
                if (texture->requires_gradients()) {
                    disabled++;
                    texture->disable_gradients();
                }
            }
        }
        if (disabled != 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "Disabled gradient computation in {} "
                "texture{} for non-is_differentiable integrator.",
                disabled, disabled > 1u ? "s" : "");
        }
    }
    return scene;
}

Scene::~Scene() noexcept = default;

}// namespace luisa::render

//
// Created by Mike on 2021/12/8.
//

#include <mutex>

#include <util/thread_pool.h>
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
#include <base/medium.h>
#include <base/phase_function.h>

namespace luisa::render {

struct Scene::Config {
    float shadow_terminator{0.f};
    float intersection_offset{0.f};
    luisa::vector<NodeHandle> internal_nodes;
    luisa::unordered_map<luisa::string, NodeHandle> nodes;
    Integrator *integrator{nullptr};
    Environment *environment{nullptr};
    Medium *environment_medium{nullptr};
    Spectrum *spectrum{nullptr};
    luisa::vector<Camera *> cameras;
    luisa::vector<Shape *> shapes;
};

const Integrator *Scene::integrator() const noexcept { return _config->integrator; }
const Environment *Scene::environment() const noexcept { return _config->environment; }
const Medium *Scene::environment_medium() const noexcept { return _config->environment_medium; }
const Spectrum *Scene::spectrum() const noexcept { return _config->spectrum; }
luisa::span<const Shape *const> Scene::shapes() const noexcept { return _config->shapes; }
luisa::span<const Camera *const> Scene::cameras() const noexcept { return _config->cameras; }
float Scene::shadow_terminator_factor() const noexcept { return _config->shadow_terminator; }
float Scene::intersection_offset_factor() const noexcept { return _config->intersection_offset; }

namespace detail {

[[nodiscard]] static auto &scene_plugin_registry() noexcept {
    static luisa::unordered_map<luisa::string, luisa::unique_ptr<DynamicModule>> registry;
    return registry;
}

[[nodiscard]] static auto &scene_plugin_registry_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] static auto &scene_plugin_load(
    const std::filesystem::path &runtime_dir, SceneNodeTag tag, luisa::string_view impl_type) noexcept {
    std::scoped_lock lock{detail::scene_plugin_registry_mutex()};
    auto name = luisa::format("luisa-render-{}-{}", scene_node_tag_description(tag), impl_type);
    for (auto &c : name) { c = static_cast<char>(std::tolower(c)); }
    auto &&registry = detail::scene_plugin_registry();
    if (auto iter = registry.find(name); iter != registry.end()) {
        return *iter->second;
    }
    auto module = luisa::make_unique<DynamicModule>(DynamicModule::load(runtime_dir, name));
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

Medium *Scene::load_medium(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Medium *>(load_node(SceneNodeTag::MEDIUM, desc));
}

PhaseFunction *Scene::load_phase_function(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<PhaseFunction *>(load_node(SceneNodeTag::PHASE_FUNCTION, desc));
}

luisa::unique_ptr<Scene> Scene::create(const Context &ctx, const SceneDesc *desc) noexcept {
    if (!desc->root()->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Root node is not defined "
            "in the scene description.");
    }
    auto scene = luisa::make_unique<Scene>(ctx);
    scene->_config->shadow_terminator = desc->root()->property_float_or_default("shadow_terminator", 0.f);
    scene->_config->intersection_offset = desc->root()->property_float_or_default("intersection_offset", 0.f);
    scene->_config->spectrum = scene->load_spectrum(desc->root()->property_node_or_default(
        "spectrum", SceneNodeDesc::shared_default_spectrum("sRGB")));
    scene->_config->integrator = scene->load_integrator(
        desc->root()->property_node("integrator"));
    scene->_config->environment = scene->load_environment(
        desc->root()->property_node_or_default("environment"));
    scene->_config->environment_medium = scene->load_medium(
        desc->root()->property_node_or_default("environment_medium"));
    auto cameras = desc->root()->property_node_list("cameras");
    auto shapes = desc->root()->property_node_list("shapes");
    auto environments = desc->root()->property_node_or_default("environments", SceneNodeDesc::shared_default_medium("Null"));
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
    global_thread_pool().synchronize();
    return scene;
}

Scene::~Scene() noexcept = default;

}// namespace luisa::render

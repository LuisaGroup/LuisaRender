//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <span>
#include <mutex>

#include <core/stl.h>
#include <core/dynamic_module.h>
#include <core/basic_types.h>
#include <runtime/context.h>
#include <base/scene_node.h>

namespace luisa::render {

using compute::Context;

class SceneDesc;
class SceneNodeDesc;

class Camera;
class Film;
class Filter;
class Integrator;
class Surface;
class Light;
class Sampler;
class Shape;
class Transform;
class LightSampler;
class Environment;
class Texture;
class TextureMapping;
class Spectrum;
class Medium;
class PhaseFunction;

class Scene {

public:
    using NodeCreater = SceneNode *(Scene *, const SceneNodeDesc *);
    using NodeDeleter = void(SceneNode *);
    using NodeHandle = luisa::unique_ptr<SceneNode, NodeDeleter *>;

    struct Config;

private:
    const Context &_context;
    luisa::unique_ptr<Config> _config;
    std::recursive_mutex _mutex;

public:
    // for internal use only, call Scene::create() instead
    explicit Scene(const Context &ctx) noexcept;
    ~Scene() noexcept;
    Scene(Scene &&scene) noexcept = delete;
    Scene(const Scene &scene) noexcept = delete;
    Scene &operator=(Scene &&scene) noexcept = delete;
    Scene &operator=(const Scene &scene) noexcept = delete;
    [[nodiscard]] SceneNode *load_node(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Camera *load_camera(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Film *load_film(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Filter *load_filter(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Integrator *load_integrator(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Surface *load_surface(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Light *load_light(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Sampler *load_sampler(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Shape *load_shape(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Transform *load_transform(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] LightSampler *load_light_sampler(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Environment *load_environment(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Texture *load_texture(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] TextureMapping *load_texture_mapping(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Spectrum *load_spectrum(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Medium *load_medium(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] PhaseFunction *load_phase_function(const SceneNodeDesc *desc) noexcept;

public:
    [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const SceneDesc *desc) noexcept;
    [[nodiscard]] const Integrator *integrator() const noexcept;
    [[nodiscard]] const Environment *environment() const noexcept;
    [[nodiscard]] const Medium *environment_medium() const noexcept;
    [[nodiscard]] const Spectrum *spectrum() const noexcept;
    [[nodiscard]] luisa::span<const Shape *const> shapes() const noexcept;
    [[nodiscard]] luisa::span<const Camera *const> cameras() const noexcept;
    [[nodiscard]] float shadow_terminator_factor() const noexcept;
    [[nodiscard]] float intersection_offset_factor() const noexcept;
};

}// namespace luisa::render

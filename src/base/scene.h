//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <span>

#include <core/hash.h>
#include <core/allocator.h>
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
class Material;
class Light;
class Sampler;
class Shape;
class Transform;
class Environment;

class Scene {

public:
    using NodeCreater = SceneNode *(Scene *, const SceneNodeDesc *);
    using NodeDeleter = void(SceneNode *);
    using NodeHandle = std::unique_ptr<SceneNode, NodeDeleter *>;

    struct Config;
    struct Data;

private:
    const Context &_context;
    luisa::unique_ptr<Config> _config;

public:
    // for internal use only, call Scene::create instead
    explicit Scene(const Context &ctx) noexcept;
    ~Scene() noexcept;
    Scene(Scene &&scene) noexcept = delete;
    Scene(const Scene &scene) noexcept = delete;
    Scene &operator=(Scene &&scene) noexcept = delete;
    Scene &operator=(const Scene &scene) noexcept = delete;
    [[nodiscard]] SceneNode *load_node(SceneNode::Tag tag, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Camera *load_camera(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Film *load_film(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Filter *load_filter(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Integrator *load_integrator(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Material *load_material(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Light *load_light(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Sampler *load_sampler(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Shape *load_shape(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Transform *load_transform(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Environment *load_environment(const SceneNodeDesc *desc) noexcept;

public:
    [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const SceneDesc *desc) noexcept;
    [[nodiscard]] const Integrator *integrator() const noexcept;
    [[nodiscard]] std::span<const Shape *const> shapes() const noexcept;
    [[nodiscard]] std::span<const Camera *const> cameras() const noexcept;
    [[nodiscard]] std::span<const Environment *const> environments() const noexcept;
    [[nodiscard]] uint spp() const noexcept;
};

}

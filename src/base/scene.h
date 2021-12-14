//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <span>

#include <core/hash.h>
#include <core/allocator.h>
#include <core/dynamic_module.h>
#include <runtime/context.h>
#include <base/scene_node.h>

namespace luisa::render {

using compute::Context;
using compute::Stream;

class SceneDesc;
class SceneNodeDesc;

class Camera;
class Film;
class Filter;
class Integrator;
class Material;
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

private:
    friend class SceneNode;
    [[nodiscard]] SceneNode *_node(SceneNode::Tag tag, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Camera *_camera(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Film *_film(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Filter *_filter(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Integrator *_integrator(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Material *_material(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Sampler *_sampler(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Shape *_shape(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Transform *_transform(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Environment *_environment(const SceneNodeDesc *desc) noexcept;

public:
    // for internal use only, call Scene::create instead
    explicit Scene(const Context &ctx) noexcept;
    ~Scene() noexcept;
    Scene(Scene &&scene) noexcept = delete;
    Scene(const Scene &scene) noexcept = delete;
    Scene &operator=(Scene &&scene) noexcept = delete;
    Scene &operator=(const Scene &scene) noexcept = delete;

public:
    [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const SceneDesc *desc) noexcept;
    [[nodiscard]] const Integrator *integrator() const noexcept;
    [[nodiscard]] std::span<const Shape *const> shapes() const noexcept;
    [[nodiscard]] std::span<const Camera *const> cameras() const noexcept;
    [[nodiscard]] std::span<const Environment *const> environments() const noexcept;
    // TODO: build & update
};

}

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
class SceneDescNode;

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
    using NodeCreater = SceneNode *(Scene *, const SceneDescNode *);
    using NodeDeleter = void(SceneNode *);
    using NodeHandle = std::unique_ptr<SceneNode, NodeDeleter *>;

private:
    const Context &_context;
    luisa::vector<NodeHandle> _internal_nodes;
    luisa::unordered_map<luisa::string, NodeHandle, Hash64> _nodes;

private:
    Integrator *_render_integrator;
    luisa::vector<Camera *> _cameras;
    luisa::vector<Shape *> _shapes;
    luisa::vector<Environment *> _environments;

    friend class SceneNode;
    [[nodiscard]] SceneNode *_node(SceneNode::Tag tag, const SceneDescNode *desc) noexcept;
    [[nodiscard]] Camera *_camera(const SceneDescNode *desc) noexcept;
    [[nodiscard]] Film *_film(const SceneDescNode *desc) noexcept;
    [[nodiscard]] Filter *_filter(const SceneDescNode *desc) noexcept;
    [[nodiscard]] Integrator *_integrator(const SceneDescNode *desc) noexcept;
    [[nodiscard]] Material *_material(const SceneDescNode *desc) noexcept;
    [[nodiscard]] Sampler *_sampler(const SceneDescNode *desc) noexcept;
    [[nodiscard]] Shape *_shape(const SceneDescNode *desc) noexcept;
    [[nodiscard]] Transform *_transform(const SceneDescNode *desc) noexcept;
    [[nodiscard]] Environment *_environment(const SceneDescNode *desc) noexcept;

public:
    // for internal use only, call Scene::create instead
    explicit Scene(const Context &ctx) noexcept;
    ~Scene() noexcept;
    Scene(Scene &&scene) noexcept = delete;
    Scene(const Scene &scene) noexcept = delete;
    Scene &operator=(Scene &&scene) noexcept = delete;
    Scene &operator=(const Scene &scene) noexcept = delete;

public:
    // public interfaces
    [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const SceneDesc *desc) noexcept;
    [[nodiscard]] auto integrator() const noexcept { return const_cast<const Integrator *>(_render_integrator); }
    [[nodiscard]] auto cameras() const noexcept { return std::span{_cameras}; }
    [[nodiscard]] auto shapes() const noexcept { return std::span{_shapes}; }
    [[nodiscard]] auto environments() const noexcept { return std::span{_environments}; }
};

}

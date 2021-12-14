//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <core/hash.h>
#include <core/allocator.h>
#include <core/dynamic_module.h>
#include <runtime/context.h>
#include <base/scene_node.h>

namespace luisa::render {

using compute::Context;
using compute::Stream;

class Scene {

public:
    using NodeDeleter = void(*)(SceneNode *);
    using NodeHandle = std::unique_ptr<SceneNode, NodeDeleter>;

private:
    using Node = SceneNode;
    const Context *_context;
    luisa::vector<NodeHandle> _internal_nodes;
    luisa::unordered_map<luisa::string, NodeHandle, Hash64> _nodes;

public:
    explicit Scene(const Context &ctx) noexcept;
    ~Scene() noexcept;
    Scene(Scene &&scene) noexcept = default;
    Scene(const Scene &scene) noexcept = delete;
    Scene &operator=(Scene &&scene) noexcept = default;
    Scene &operator=(const Scene &scene) noexcept = delete;
    [[nodiscard]] Node *node(Node::Tag tag, const SceneDescNode *desc) noexcept;
};

}

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

private:
    using Node = SceneNode;
    const Context *_context;
    luisa::unordered_map<luisa::string, std::unique_ptr<Node, void(*)(Node *)>, Hash64> _nodes;

public:
    explicit Scene(const Context &ctx) noexcept;
    ~Scene() noexcept;
    Scene(Scene &&scene) noexcept = default;
    Scene(const Scene &scene) noexcept = delete;
    Scene &operator=(Scene &&scene) noexcept = default;
    Scene &operator=(const Scene &scene) noexcept = delete;
    [[nodiscard]] Node *add(Node::Tag tag, std::string_view identifier, std::string_view impl_type) noexcept;
};

}

//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <core/hash.h>
#include <core/allocator.h>
#include <core/dynamic_module.h>
#include <runtime/context.h>

namespace luisa::render {

using compute::Context;

class Scene {

public:
    class Node {

    public:
        enum struct Tag : uint32_t {
            CAMERA,
            SHAPE,
            MATERIAL,
            TRANSFORM,
            FILM,
            FILTER,
            SAMPLER,
            INTEGRATOR
        };

    private:
        Tag _tag;

    public:
        explicit Node(Tag tag) noexcept: _tag{tag} {}
        virtual ~Node() noexcept = default;
        [[nodiscard]] auto tag() const noexcept { return _tag; }
    };

private:
    const Context *_context;
    luisa::unordered_map<luisa::string, std::unique_ptr<Node, void(*)(Node *)>, Hash64> _nodes;

private:
    [[nodiscard]] DynamicModule &_load_plugin(Node::Tag tag, std::string_view impl_type) noexcept;

public:
    Scene(const Context &ctx) noexcept;
    ~Scene() noexcept;
    Scene(Scene &&scene) noexcept = default;
    Scene(const Scene &scene) noexcept = delete;
    Scene &operator=(Scene &&scene) noexcept = default;
    Scene &operator=(const Scene &scene) noexcept = delete;
    [[nodiscard]] Node *add(Node::Tag tag, std::string_view identifier, std::string_view impl_type) noexcept;
};

}

//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <cstddef>
#include <string_view>

#include <core/basic_types.h>
#include <dsl/syntax.h>
#include <sdl/scene_node_desc.h>

namespace luisa::compute {
class Device;
class Stream;
class CommandBuffer;
}// namespace luisa::compute

namespace luisa::render {

using compute::Device;
using compute::Stream;

using compute::Expr;
using compute::Float;
using compute::Float2;
using compute::Float3;
using compute::Float4;
using compute::Var;

class Scene;
class Pipeline;

class SceneNode {

public:
    using Tag = SceneNodeTag;

private:
    intptr_t _scene : 56u;
    Tag _tag : 8u;

public:
    SceneNode(const Scene *scene, const SceneNodeDesc *desc, Tag tag) noexcept;
    SceneNode(SceneNode &&) noexcept = delete;
    SceneNode(const SceneNode &) noexcept = delete;
    SceneNode &operator=(SceneNode &&) noexcept = delete;
    SceneNode &operator=(const SceneNode &) noexcept = delete;
    virtual ~SceneNode() noexcept = default;
    [[nodiscard]] auto scene() const noexcept { return reinterpret_cast<const Scene *>(_scene); }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] virtual luisa::string_view impl_type() const noexcept = 0;
};

}// namespace luisa::render

#define LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(cls)                   \
    LUISA_EXPORT_API luisa::render::SceneNode *create(             \
        luisa::render::Scene *scene,                               \
        const luisa::render::SceneNodeDesc *desc) LUISA_NOEXCEPT { \
        return luisa::new_with_allocator<cls>(scene, desc);        \
    }                                                              \
    LUISA_EXPORT_API void destroy(                                 \
        luisa::render::SceneNode *node) LUISA_NOEXCEPT {           \
        luisa::delete_with_allocator(node);                        \
    }

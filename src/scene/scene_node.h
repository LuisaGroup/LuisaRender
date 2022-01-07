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
}// namespace luisa::compute

namespace luisa::render {

using compute::Device;
using compute::Stream;

using compute::Expr;
using compute::Var;
using compute::Float;
using compute::Float2;
using compute::Float3;
using compute::Float4;

class Scene;
class Pipeline;

class SceneNode {

public:
    using Tag = SceneNodeTag;

    class Instance {

    protected:
        Instance() noexcept = default;
        ~Instance() noexcept = default;

    public:
        Instance(Instance &&) noexcept = delete;
        Instance(const Instance &) noexcept = delete;
        Instance &operator=(Instance &&) noexcept = delete;
        Instance &operator=(const Instance &) noexcept = delete;
    };

private:
    const Scene *_scene;
    Tag _tag;

public:
    SceneNode(const Scene *scene, const SceneNodeDesc *desc, Tag tag) noexcept;
    SceneNode(SceneNode &&) noexcept = delete;
    SceneNode(const SceneNode &) noexcept = delete;
    SceneNode &operator=(SceneNode &&) noexcept = delete;
    SceneNode &operator=(const SceneNode &) noexcept = delete;
    virtual ~SceneNode() noexcept = default;
    [[nodiscard]] auto scene() const noexcept { return _scene; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] virtual std::string_view impl_type() const noexcept = 0;
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

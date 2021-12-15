//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <cstddef>
#include <string_view>
#include <dsl/syntax.h>

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
class SceneNodeDesc;
class Pipeline;

class SceneNode {

public:
    enum struct Tag : uint32_t {
        ROOT,
        INTERNAL,
        CAMERA,
        SHAPE,
        MATERIAL,
        LIGHT,
        TRANSFORM,
        FILM,
        FILTER,
        SAMPLER,
        INTEGRATOR,
        ENVIRONMENT
        // TODO: MEDIUM?
    };

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
    [[nodiscard]] static constexpr std::string_view tag_description(Tag tag) noexcept;
};

constexpr std::string_view SceneNode::tag_description(SceneNode::Tag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case SceneNode::Tag::ROOT: return "__root__"sv;
        case SceneNode::Tag::INTERNAL: return "__internal__"sv;
        case SceneNode::Tag::CAMERA: return "Camera"sv;
        case SceneNode::Tag::SHAPE: return "Shape"sv;
        case SceneNode::Tag::MATERIAL: return "Material"sv;
        case SceneNode::Tag::LIGHT: return "Light"sv;
        case SceneNode::Tag::TRANSFORM: return "Transform"sv;
        case SceneNode::Tag::FILM: return "Film"sv;
        case SceneNode::Tag::FILTER: return "Filter"sv;
        case SceneNode::Tag::SAMPLER: return "Sampler"sv;
        case SceneNode::Tag::INTEGRATOR: return "Integrator"sv;
        case SceneNode::Tag::ENVIRONMENT: return "Environment"sv;
        default: break;
    }
    return "__invalid__"sv;
}

}// namespace luisa::render

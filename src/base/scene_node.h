//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <cstddef>
#include <string_view>

namespace luisa::compute {
class Device;
class CommandBuffer;
}

namespace luisa::render {

using compute::Device;
using compute::CommandBuffer;

class Scene;
class SceneNodeDesc;

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

private:
    Scene *_scene;
    Tag _tag;

public:
    SceneNode(Scene *scene, const SceneNodeDesc *desc, Tag tag) noexcept;
    virtual ~SceneNode() noexcept = default;
    [[nodiscard]] auto scene() noexcept { return _scene; }
    [[nodiscard]] auto scene() const noexcept { return const_cast<const Scene *>(_scene); }
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

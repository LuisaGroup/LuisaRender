//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <cstddef>
#include <string_view>

namespace luisa::compute {
class CommandBuffer;
}

namespace luisa::render {

using compute::CommandBuffer;

class SceneDescNode;

class SceneNode {

public:
    enum struct Tag : uint32_t {
        ROOT,
        INTERNAL,
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
    explicit SceneNode(Tag tag) noexcept : _tag{tag} {}
    virtual ~SceneNode() noexcept = default;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] virtual std::string_view impl_type() const noexcept = 0;
    [[nodiscard]] virtual size_t child_count() const noexcept = 0;
    [[nodiscard]] virtual SceneNode *child(size_t index) noexcept = 0;
    [[nodiscard]] virtual const SceneNode *child(size_t index) const noexcept;
    virtual void load(const SceneDescNode &desc) noexcept = 0;
    virtual void dump(SceneDescNode &desc) const noexcept = 0;
    virtual void build(CommandBuffer &command_buffer) noexcept = 0;
    [[nodiscard]] static constexpr std::string_view tag_description(Tag tag) noexcept;
};

constexpr std::string_view SceneNode::tag_description(SceneNode::Tag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case SceneNode::Tag::ROOT: return "root"sv;
        case SceneNode::Tag::INTERNAL: return "internal"sv;
        case SceneNode::Tag::CAMERA: return "camera"sv;
        case SceneNode::Tag::SHAPE: return "shape"sv;
        case SceneNode::Tag::MATERIAL: return "material"sv;
        case SceneNode::Tag::TRANSFORM: return "transform"sv;
        case SceneNode::Tag::FILM: return "film"sv;
        case SceneNode::Tag::FILTER: return "filter"sv;
        case SceneNode::Tag::SAMPLER: return "sampler"sv;
        case SceneNode::Tag::INTEGRATOR: return "integrator"sv;
        default: break;
    }
    return "invalid"sv;
}

}// namespace luisa::render

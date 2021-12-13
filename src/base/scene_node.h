//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <cstddef>
#include <string_view>

#include <base/scene_desc.h>

namespace luisa::compute {
class CommandBuffer;
}

namespace luisa::render {

using compute::CommandBuffer;

class SceneNode {

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
    explicit SceneNode(Tag tag) noexcept: _tag{tag} {}
    virtual ~SceneNode() noexcept = default;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] virtual std::string_view impl_type() const noexcept = 0;
    [[nodiscard]] virtual size_t child_count() const noexcept = 0;
    [[nodiscard]] virtual SceneNode *child(size_t index) noexcept = 0;
    [[nodiscard]] virtual const SceneNode *child(size_t index) const noexcept;
    virtual void load(const SceneDesc::Node &desc) noexcept = 0;
    virtual void dump(SceneDesc::Node &desc) const noexcept = 0;
    virtual void build(CommandBuffer &command_buffer) noexcept = 0;
};

}

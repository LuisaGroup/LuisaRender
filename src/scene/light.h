//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <runtime/bindless_array.h>
#include <scene/scene_node.h>

namespace luisa::render {

using compute::BindlessArray;

class Shape;

struct LightInstance {
    uint instance_id;
    uint light_buffer_id_and_tag;
    uint position_buffer_id_and_offset;
    uint attribute_buffer_id_and_offset;
};

class Light : public SceneNode {

public:
    static constexpr auto property_flag_black = 1u;
    static constexpr auto property_flag_two_sided = 2u;
    static constexpr auto property_flag_uniform = 4u;

public:
    struct Evaluation {

    };

    struct Sample {

    };

    struct Context {

    };

    class Interface {

    public:
        [[nodiscard]] virtual luisa::unique_ptr<Context> decode(const BindlessArray &array, uint buffer_id) const noexcept = 0;
        virtual ~Interface() noexcept = default;
    };

public:
    Light(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual float power(const Shape *shape) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Interface> create_interface() const noexcept = 0;
    [[nodiscard]] virtual uint property_flags() const noexcept = 0;
    [[nodiscard]] virtual uint /* bindless buffer id */ encode(Pipeline &pipeline, CommandBuffer &command_buffer, const Shape *shape) const noexcept = 0;
};

}

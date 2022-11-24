//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <rtx/mesh.h>
#include <util/half.h>
#include <util/vertex.h>
#include <base/scene_node.h>

namespace luisa::render {

class Light;
class Surface;

using compute::AccelUsageHint;
using compute::Triangle;

class Light;
class Surface;
class Transform;

class Shape : public SceneNode {

public:
    class Handle;

public:
    static constexpr auto property_flag_has_vertex_normal = 1u << 0u;
    static constexpr auto property_flag_has_vertex_tangent = 1u << 1u;
    static constexpr auto property_flag_has_vertex_uv = 1u << 2u;
    static constexpr auto property_flag_has_vertex_color = 1u << 3u;
    static constexpr auto property_flag_has_surface = 1u << 4u;
    static constexpr auto property_flag_has_light = 1u << 5u;
    static constexpr auto property_flag_has_medium = 1u << 6u;

private:
    const Surface *_surface;
    const Light *_light;
    const Transform *_transform;

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] const Surface *surface() const noexcept;
    [[nodiscard]] const Light *light() const noexcept;
    [[nodiscard]] const Transform *transform() const noexcept;
    [[nodiscard]] virtual bool visible() const noexcept;
    [[nodiscard]] virtual float shadow_terminator_factor() const noexcept;
    [[nodiscard]] virtual float intersection_offset_factor() const noexcept;
    [[nodiscard]] virtual bool is_mesh() const noexcept;
    [[nodiscard]] virtual uint vertex_properties() const noexcept;
    [[nodiscard]] bool has_vertex_normal() const noexcept;
    [[nodiscard]] bool has_vertex_uv() const noexcept;
    [[nodiscard]] bool has_vertex_tangent() const noexcept;
    [[nodiscard]] bool has_vertex_color() const noexcept;
    [[nodiscard]] virtual luisa::span<const Vertex> vertices() const noexcept;      // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const Triangle> triangles() const noexcept;   // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const Shape *const> children() const noexcept;// empty if the shape is a mesh
    [[nodiscard]] virtual bool deformable() const noexcept;                         // true if the shape will not deform
    [[nodiscard]] virtual AccelUsageHint build_hint() const noexcept;               // accel struct build quality, only considered for meshes
};

template<typename BaseShape>
class ShadowTerminatorShapeWrapper : public BaseShape {

private:
    float _shadow_terminator;

public:
    ShadowTerminatorShapeWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseShape{scene, desc},
          _shadow_terminator{std::clamp(desc->property_float_or_default("shadow_terminator", 0.f), 0.f, 1.f)} {}
    [[nodiscard]] float shadow_terminator_factor() const noexcept override { return _shadow_terminator; }
};

template<typename BaseShape>
class IntersectionOffsetShapeWrapper : public BaseShape {

private:
    float _intersection_offset;

public:
    IntersectionOffsetShapeWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseShape{scene, desc},
          _intersection_offset{std::clamp(desc->property_float_or_default("intersection_offset", 0.f), 0.f, 1.f)} {}
    [[nodiscard]] float intersection_offset_factor() const noexcept override { return _intersection_offset; }
};

template<typename BaseShape>
class VisibilityShapeWrapper : public BaseShape {

private:
    bool _visible;

public:
    VisibilityShapeWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseShape{scene, desc},
          _visible{desc->property_bool_or_default("visible", true)} {}
    [[nodiscard]] bool visible() const noexcept override { return _visible; }
};

struct alignas(16) Shape::Handle {

    static constexpr auto property_flag_bits = 10u;
    static constexpr auto property_flag_mask = (1u << property_flag_bits) - 1u;

    static constexpr auto buffer_base_max = (1u << (32u - property_flag_bits)) - 1u;

    static constexpr auto light_tag_bits = 16u;
    static constexpr auto surface_tag_bits = 32u - light_tag_bits;
    static constexpr auto surface_tag_max = (1u << surface_tag_bits) - 1u;
    static constexpr auto light_tag_mask = (1u << light_tag_bits) - 1u;

    static constexpr auto vertex_buffer_id_offset = 0u;
    static constexpr auto triangle_buffer_id_offset = 1u;
    static constexpr auto alias_table_buffer_id_offset = 2u;
    static constexpr auto pdf_buffer_id_offset = 3u;

    uint buffer_base_and_properties;
    uint surface_tag_and_light_tag;
    uint triangle_buffer_size;
    uint shadow_term_and_intersection_offset;

    [[nodiscard]] static Shape::Handle encode(uint buffer_base, uint flags,
                                              uint surface_tag, uint light_tag, uint tri_count,
                                              float shadow_terminator, float intersection_offset) noexcept;
};

static_assert(sizeof(Shape::Handle) == 16u);

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::Shape::Handle,
    buffer_base_and_properties,
    surface_tag_and_light_tag,
    triangle_buffer_size,
    shadow_term_and_intersection_offset) {

private:
    [[nodiscard]] static auto _decode_fixed_point(auto x) noexcept {
        constexpr auto fixed_point_bits = 16u;
        constexpr auto fixed_point_mask = (1u << fixed_point_bits) - 1u;
        constexpr auto fixed_point_scale = 1.0f / static_cast<float>(1u << fixed_point_bits);
        return cast<float>(x & fixed_point_mask) * fixed_point_scale;
    }

public:
    [[nodiscard]] auto geometry_buffer_base() const noexcept { return buffer_base_and_properties >> luisa::render::Shape::Handle::property_flag_bits; }
    [[nodiscard]] auto property_flags() const noexcept { return buffer_base_and_properties & luisa::render::Shape::Handle::property_flag_mask; }
    [[nodiscard]] auto vertex_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::vertex_buffer_id_offset; }
    [[nodiscard]] auto triangle_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::triangle_buffer_id_offset; }
    [[nodiscard]] auto triangle_count() const noexcept { return triangle_buffer_size; }
    [[nodiscard]] auto alias_table_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::alias_table_buffer_id_offset; }
    [[nodiscard]] auto pdf_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::pdf_buffer_id_offset; }
    [[nodiscard]] auto surface_tag() const noexcept { return surface_tag_and_light_tag >> luisa::render::Shape::Handle::light_tag_bits; }
    [[nodiscard]] auto light_tag() const noexcept { return surface_tag_and_light_tag & luisa::render::Shape::Handle::light_tag_mask; }
    [[nodiscard]] auto test_property_flag(luisa::uint flag) const noexcept { return (property_flags() & flag) != 0u; }
    [[nodiscard]] auto has_vertex_normal() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_vertex_normal); }
    [[nodiscard]] auto has_vertex_tangent() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_vertex_tangent); }
    [[nodiscard]] auto has_vertex_color() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_vertex_color); }
    [[nodiscard]] auto has_vertex_uv() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_vertex_uv); }
    [[nodiscard]] auto has_light() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_light); }
    [[nodiscard]] auto has_surface() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_surface); }
    [[nodiscard]] auto has_medium() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_medium); }
    [[nodiscard]] auto shadow_terminator_factor() const noexcept { return _decode_fixed_point(shadow_term_and_intersection_offset >> 16u); }
    [[nodiscard]] auto intersection_offset_factor() const noexcept { return _decode_fixed_point(shadow_term_and_intersection_offset & 0xffffu); }
};
// clang-format on

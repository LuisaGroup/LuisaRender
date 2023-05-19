//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <runtime/rtx/mesh.h>
#include <util/half.h>
#include <util/vertex.h>
#include <base/scene_node.h>
#include <base/scene.h>

namespace luisa::render {

class Light;
class Surface;
class Transform;
class Medium;

using compute::AccelOption;
using compute::Triangle;

struct MeshView {
    luisa::span<const Vertex> vertices;
    luisa::span<const Triangle> triangles;
};

class Shape : public SceneNode {

public:
    class Handle;

public:
    static constexpr auto property_flag_has_vertex_normal = 1u << 0u;
    static constexpr auto property_flag_has_vertex_uv = 1u << 1u;
    static constexpr auto property_flag_has_surface = 1u << 2u;
    static constexpr auto property_flag_has_light = 1u << 3u;
    static constexpr auto property_flag_has_medium = 1u << 4u;

private:
    const Surface *_surface;
    const Light *_light;
    const Medium *_medium;
    const Transform *_transform;

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] const Surface *surface() const noexcept;
    [[nodiscard]] const Light *light() const noexcept;
    [[nodiscard]] const Medium *medium() const noexcept;
    [[nodiscard]] const Transform *transform() const noexcept;
    [[nodiscard]] virtual bool visible() const noexcept;
    [[nodiscard]] virtual float shadow_terminator_factor() const noexcept;
    [[nodiscard]] virtual float intersection_offset_factor() const noexcept;
    [[nodiscard]] virtual bool is_mesh() const noexcept;
    [[nodiscard]] virtual uint vertex_properties() const noexcept;
    [[nodiscard]] bool has_vertex_normal() const noexcept;
    [[nodiscard]] bool has_vertex_uv() const noexcept;
    [[nodiscard]] virtual MeshView mesh() const noexcept;                           // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const Shape *const> children() const noexcept;// empty if the shape is a mesh
    [[nodiscard]] virtual bool deformable() const noexcept;                         // true if the shape will not deform
    [[nodiscard]] virtual AccelOption build_option() const noexcept;                // accel struct build quality, only considered for meshes
};

template<typename BaseShape>
class ShadowTerminatorShapeWrapper : public BaseShape {

private:
    float _shadow_terminator;

public:
    ShadowTerminatorShapeWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseShape{scene, desc},
          _shadow_terminator{std::clamp(
              desc->property_float_or_default(
                  "shadow_terminator",
                  scene->shadow_terminator_factor()),
              0.f, 1.f)} {}
    [[nodiscard]] float shadow_terminator_factor() const noexcept override {
        return _shadow_terminator;
    }
};

template<typename BaseShape>
class IntersectionOffsetShapeWrapper : public BaseShape {

private:
    float _intersection_offset;

public:
    IntersectionOffsetShapeWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseShape{scene, desc},
          _intersection_offset{std::clamp(
              desc->property_float_or_default(
                  "intersection_offset",
                  scene->intersection_offset_factor()),
              0.f, 1.f)} {}
    [[nodiscard]] float intersection_offset_factor() const noexcept override {
        return _intersection_offset;
    }
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

using compute::Expr;
using compute::Float;
using compute::UInt;

class Shape::Handle {

public:
    static constexpr auto property_flag_bits = 10u;
    static constexpr auto property_flag_mask = (1u << property_flag_bits) - 1u;

    static constexpr auto buffer_base_max = (1u << (32u - property_flag_bits)) - 1u;

    static constexpr auto light_tag_bits = 12u;
    static constexpr auto surface_tag_bits = 12u;
    static constexpr auto medium_tag_bits = 32u - light_tag_bits - surface_tag_bits;
    static constexpr auto surface_tag_max = (1u << surface_tag_bits) - 1u;
    static constexpr auto light_tag_max = (1u << light_tag_bits) - 1u;
    static constexpr auto medium_tag_max = (1u << medium_tag_bits) - 1u;
    static constexpr auto light_tag_offset = 0u;
    static constexpr auto surface_tag_offset = light_tag_offset + light_tag_bits;
    static constexpr auto medium_tag_offset = surface_tag_offset + surface_tag_bits;

    static constexpr auto vertex_buffer_id_offset = 0u;
    static constexpr auto triangle_buffer_id_offset = 1u;
    static constexpr auto alias_table_buffer_id_offset = 2u;
    static constexpr auto pdf_buffer_id_offset = 3u;

private:
    UInt _buffer_base;
    UInt _properties;
    UInt _surface_tag;
    UInt _light_tag;
    UInt _medium_tag;
    UInt _triangle_count;
    Float _shadow_terminator;
    Float _intersection_offset;

private:
    Handle(Expr<uint> buffer_base, Expr<uint> flags,
           Expr<uint> surface_tag, Expr<uint> light_tag, Expr<uint> medium_tag,
           Expr<uint> triangle_count, Expr<float> shadow_terminator, Expr<float> intersection_offset) noexcept
        : _buffer_base{buffer_base}, _properties{flags},
          _surface_tag{surface_tag}, _light_tag{light_tag}, _medium_tag{medium_tag},
          _triangle_count{triangle_count},
          _shadow_terminator{shadow_terminator},
          _intersection_offset{intersection_offset} {}

public:
    Handle() noexcept = default;
    [[nodiscard]] static uint4 encode(uint buffer_base, uint flags,
                                      uint surface_tag, uint light_tag, uint medium_tag,
                                      uint tri_count, float shadow_terminator, float intersection_offset) noexcept;
    [[nodiscard]] static Shape::Handle decode(Expr<uint4> compressed) noexcept;

public:
    [[nodiscard]] auto geometry_buffer_base() const noexcept { return _buffer_base; }
    [[nodiscard]] auto property_flags() const noexcept { return _properties; }
    [[nodiscard]] auto vertex_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::vertex_buffer_id_offset; }
    [[nodiscard]] auto triangle_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::triangle_buffer_id_offset; }
    [[nodiscard]] auto triangle_count() const noexcept { return _triangle_count; }
    [[nodiscard]] auto alias_table_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::alias_table_buffer_id_offset; }
    [[nodiscard]] auto pdf_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::pdf_buffer_id_offset; }
    [[nodiscard]] auto surface_tag() const noexcept { return _surface_tag; }
    [[nodiscard]] auto light_tag() const noexcept { return _light_tag; }
    [[nodiscard]] auto medium_tag() const noexcept { return _medium_tag; }
    [[nodiscard]] auto test_property_flag(luisa::uint flag) const noexcept { return (property_flags() & flag) != 0u; }
    [[nodiscard]] auto has_vertex_normal() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_vertex_normal); }
    [[nodiscard]] auto has_vertex_uv() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_vertex_uv); }
    [[nodiscard]] auto has_light() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_light); }
    [[nodiscard]] auto has_surface() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_surface); }
    [[nodiscard]] auto has_medium() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_medium); }
    [[nodiscard]] auto shadow_terminator_factor() const noexcept { return _shadow_terminator; }
    [[nodiscard]] auto intersection_offset_factor() const noexcept { return _intersection_offset; }
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Shape::Handle)

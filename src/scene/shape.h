//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <rtx/mesh.h>
#include <scene/scene_node.h>

namespace luisa::render {

class Light;
class Material;

struct alignas(16) VertexAttribute {

    uint compressed_normal;
    uint compressed_tangent;
    float compressed_uv[2];

    [[nodiscard]] static auto encode(float3 normal, float3 tangent, float2 uv) noexcept {
        constexpr auto oct_encode = [](float3 n) noexcept {
            constexpr auto oct_wrap = [](float2 v) noexcept {
                return (1.0f - abs(v.yx())) * select(make_float2(-1.0f), make_float2(1.0f), v >= 0.0f);
            };
            auto p = n.xy() * (1.0f / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z)));
            p = n.z >= 0.0f ? p : oct_wrap(p);// in [-1, 1]
            auto u = make_uint2(clamp(round((p * 0.5f + 0.5f) * 65535.0f), 0.0f, 65535.0f));
            return u.x | (u.y << 16u);
        };
        return VertexAttribute{
            .compressed_normal = oct_encode(normal),
            .compressed_tangent = oct_encode(tangent),
            .compressed_uv = {uv.x, uv.y}};
    };
};

struct alignas(16) InstancedShape {

    static constexpr auto instance_buffer_id_shift = 12u;
    static constexpr auto instance_buffer_offset_mask = (1u << instance_buffer_id_shift) - 1u;

    static constexpr auto property_flag_bits = 8u;
    static constexpr auto property_flag_mask = (1u << property_flag_bits) - 1u;

    static constexpr auto material_buffer_id_shift = 8u;
    static constexpr auto light_buffer_id_shift = 8u;
    static constexpr auto material_tag_mask = (1u << material_buffer_id_shift) - 1u;
    static constexpr auto light_tag_mask = (1u << light_buffer_id_shift) - 1u;

    static constexpr auto position_buffer_id_offset = 0u;
    static constexpr auto attribute_buffer_id_offset = 1u;
    static constexpr auto triangle_buffer_id_offset = 2u;
    static constexpr auto alias_table_buffer_id_offset = 3u;
    static constexpr auto pdf_buffer_id_offset = 4u;

    uint buffer_id_base;
    uint properties;
    uint material_buffer_id_and_tag;
    uint light_buffer_id_and_tag;

    [[nodiscard]] static auto encode_material_buffer_id_and_tag(uint buffer_id, uint tag) noexcept {
        if (tag != ~0u && tag > material_tag_mask) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid material tag: {}.", tag);
        }
        return (buffer_id << material_buffer_id_shift) | tag;
    }

    [[nodiscard]] static auto encode_light_buffer_id_and_tag(uint buffer_id, uint tag) noexcept {
        if (tag != ~0u && tag > light_tag_mask) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid light tag: {}.", tag);
        }
        return (buffer_id << light_buffer_id_shift) | tag;
    }
};

static_assert(sizeof(InstancedShape) == 16u);

using compute::AccelBuildHint;
using compute::Triangle;

class Light;
class Material;
class Transform;

class Shape : public SceneNode {

public:
    static constexpr auto property_flag_two_sided = 1u << 0u;
    static constexpr auto property_flag_has_material = 1u << 1u;
    static constexpr auto property_flag_has_light = 1u << 2u;

private:
    const Material *_material;
    const Light *_light;
    const Transform *_transform;
    luisa::optional<bool> _two_sided;
    AccelBuildHint _build_hint{AccelBuildHint::FAST_TRACE};

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto material() const noexcept { return _material; }
    [[nodiscard]] auto light() const noexcept { return _light; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] auto build_hint() const noexcept { return _build_hint; }
    [[nodiscard]] auto two_sided() const noexcept { return _two_sided; }
    [[nodiscard]] virtual bool is_mesh() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const float3> positions() const noexcept = 0;          // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const VertexAttribute> attributes() const noexcept = 0;// empty if the shape is not a mesh or the mesh has no attributes
    [[nodiscard]] virtual luisa::span<const Triangle> triangles() const noexcept = 0;        // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const Shape *const> children() const noexcept = 0;     // empty if the shape is a mesh
    [[nodiscard]] virtual bool deformable() const noexcept = 0;                              // true if the shape will not deform
};

}// namespace luisa::render

// clang-format off

LUISA_STRUCT(
    luisa::render::VertexAttribute,
    compressed_normal,
    compressed_tangent,
    compressed_uv) {

    [[nodiscard]] static auto oct_decode(luisa::compute::Expr<luisa::uint> u) noexcept {
        using namespace luisa::compute;
        auto p = make_float2(
            cast<float>((u & 0xffffu) * (1.0f / 65535.0f)),
            cast<float>((u >> 16u) * (1.0f / 65535.0f)));
        p = p * 2.0f - 1.0f;// map to [-1, 1]
        auto n = make_float3(p, 1.0f - abs(p.x) - abs(p.y));
        auto t = saturate(-n.z);
        return normalize(make_float3(n.xy() + select(t, -t, n.xy() >= 0.0f), n.z));
    }

    [[nodiscard]] auto normal() const noexcept { return oct_decode(compressed_normal); }
    [[nodiscard]] auto tangent() const noexcept { return oct_decode(compressed_tangent); }
    [[nodiscard]] auto uv() const noexcept { return luisa::compute::dsl::make_float2(compressed_uv[0], compressed_uv[1]); }
};

LUISA_STRUCT(
    luisa::render::InstancedShape,

    buffer_id_base,
    properties,
    material_buffer_id_and_tag,
    light_buffer_id_and_tag) {

    [[nodiscard]] auto position_buffer_id() const noexcept { return buffer_id_base + luisa::render::InstancedShape::position_buffer_id_offset; }
    [[nodiscard]] auto attribute_buffer_id() const noexcept { return buffer_id_base + luisa::render::InstancedShape::attribute_buffer_id_offset; }
    [[nodiscard]] auto triangle_buffer_id() const noexcept { return buffer_id_base + luisa::render::InstancedShape::triangle_buffer_id_offset; }
    [[nodiscard]] auto alias_table_buffer_id() const noexcept { return buffer_id_base + luisa::render::InstancedShape::alias_table_buffer_id_offset; }
    [[nodiscard]] auto pdf_buffer_id() const noexcept { return buffer_id_base + luisa::render::InstancedShape::pdf_buffer_id_offset; }
    [[nodiscard]] auto material_tag() const noexcept { return material_buffer_id_and_tag & luisa::render::InstancedShape::material_tag_mask; }
    [[nodiscard]] auto material_buffer_id() const noexcept { return material_buffer_id_and_tag >> luisa::render::InstancedShape::material_buffer_id_shift; }
    [[nodiscard]] auto light_tag() const noexcept { return light_buffer_id_and_tag & luisa::render::InstancedShape::light_tag_mask; }
    [[nodiscard]] auto light_buffer_id() const noexcept { return light_buffer_id_and_tag >> luisa::render::InstancedShape::light_buffer_id_shift; }
    [[nodiscard]] auto property_flags() const noexcept { return properties & luisa::render::InstancedShape::property_flag_mask; }
    [[nodiscard]] auto test_property_flag(luisa::uint flag) const noexcept { return (property_flags() & flag) != 0u; }
    [[nodiscard]] auto two_sided() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_two_sided); }
    [[nodiscard]] auto has_light() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_light); }
    [[nodiscard]] auto has_material() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_material); }
};

// clang-format on

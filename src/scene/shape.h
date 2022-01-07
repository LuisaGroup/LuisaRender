//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <rtx/mesh.h>
#include <scene/material.h>
#include <scene/light.h>

namespace luisa::render {

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

struct alignas(16) MeshInstance {

    static constexpr auto instance_buffer_id_shift = 12u;
    static constexpr auto instance_buffer_offset_mask = (1u << instance_buffer_id_shift) - 1u;

    static constexpr auto position_buffer_id_shift = 13u;
    static constexpr auto attribute_buffer_id_shift = 13u;
    static constexpr auto triangle_buffer_id_shift = 13u;
    static constexpr auto area_cdf_buffer_id_shift = 13u;

    static constexpr auto position_buffer_element_alignment = 16u;
    static constexpr auto attribute_buffer_element_alignment = 16u;
    static constexpr auto triangle_buffer_element_alignment = 16u;
    static constexpr auto area_cdf_buffer_element_alignment = 16u;

    static constexpr auto position_buffer_offset_mask = (1u << position_buffer_id_shift) - 1u;
    static constexpr auto attribute_buffer_offset_mask = (1u << attribute_buffer_id_shift) - 1u;
    static constexpr auto triangle_buffer_offset_mask = (1u << triangle_buffer_id_shift) - 1u;
    static constexpr auto area_cdf_buffer_offset_mask = (1u << area_cdf_buffer_id_shift) - 1u;

    static constexpr auto material_buffer_id_shift = 12u;
    static constexpr auto light_buffer_id_shift = 12u;
    static constexpr auto material_tag_mask = (1u << material_buffer_id_shift) - 1u;
    static constexpr auto light_tag_mask = (1u << light_buffer_id_shift) - 1u;

    // vertices & indices
    uint position_buffer_id_and_offset;// (buffer_id << shift) | offset
    uint attribute_buffer_id_and_offset;
    uint triangle_buffer_id_and_offset;
    uint triangle_buffer_size;

    // other info
    uint material_and_light_property_flags;
    uint material_buffer_id_and_tag;// = (buffer_id << shift) | tag
    uint light_buffer_id_and_tag;   // = (buffer_id << shift) | tag
    uint area_cdf_buffer_id_and_offset;
};

static_assert(sizeof(MeshInstance) == 32);

using compute::AccelBuildHint;
using compute::Triangle;

class Light;
class Material;
class Transform;

class Shape : public SceneNode {

private:
    const Material *_material;
    const Light *_light;
    const Transform *_transform;
    AccelBuildHint _build_hint{AccelBuildHint::FAST_TRACE};

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto material() const noexcept { return _material; }
    [[nodiscard]] auto light() const noexcept { return _light; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] auto build_hint() const noexcept { return _build_hint; }
    [[nodiscard]] virtual bool is_mesh() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const float3> positions() const noexcept = 0;           // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const VertexAttribute> attributes() const noexcept = 0; // empty if the shape is not a mesh or the mesh has no attributes
    [[nodiscard]] virtual luisa::span<const Triangle> triangles() const noexcept = 0;         // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const Shape *const> children() const noexcept = 0;// empty if the shape is a mesh
    [[nodiscard]] virtual bool is_rigid() const noexcept = 0;                           // true if the shape will not deform
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
    luisa::render::MeshInstance,

    position_buffer_id_and_offset,
    attribute_buffer_id_and_offset,
    triangle_buffer_id_and_offset,
    triangle_buffer_size,
    material_and_light_property_flags,
    material_buffer_id_and_tag,
    light_buffer_id_and_tag,
    area_cdf_buffer_id_and_offset) {

    [[nodiscard]] auto position_buffer_id() const noexcept { return position_buffer_id_and_offset >> luisa::render::MeshInstance::position_buffer_id_shift; }
    [[nodiscard]] auto position_buffer_offset() const noexcept { return (position_buffer_id_and_offset & luisa::render::MeshInstance::position_buffer_offset_mask) * luisa::render::MeshInstance::position_buffer_element_alignment; }
    [[nodiscard]] auto attribute_buffer_id() const noexcept { return attribute_buffer_id_and_offset >> luisa::render::MeshInstance::attribute_buffer_id_shift; }
    [[nodiscard]] auto attribute_buffer_offset() const noexcept { return (attribute_buffer_id_and_offset & luisa::render::MeshInstance::attribute_buffer_offset_mask) * luisa::render::MeshInstance::attribute_buffer_element_alignment; }
    [[nodiscard]] auto triangle_buffer_id() const noexcept { return triangle_buffer_id_and_offset >> luisa::render::MeshInstance::triangle_buffer_id_shift; }
    [[nodiscard]] auto triangle_buffer_offset() const noexcept { return (triangle_buffer_id_and_offset & luisa::render::MeshInstance::triangle_buffer_offset_mask) * luisa::render::MeshInstance::triangle_buffer_element_alignment; }
    [[nodiscard]] auto triangle_count() const noexcept { return triangle_buffer_size; }
    [[nodiscard]] auto area_cdf_buffer_id() const noexcept { return area_cdf_buffer_id_and_offset >> luisa::render::MeshInstance::area_cdf_buffer_id_shift; }
    [[nodiscard]] auto area_cdf_buffer_offset() const noexcept { return (area_cdf_buffer_id_and_offset & luisa::render::MeshInstance::area_cdf_buffer_offset_mask) * luisa::render::MeshInstance::area_cdf_buffer_element_alignment; }
    [[nodiscard]] auto material_tag() const noexcept { return material_buffer_id_and_tag & luisa::render::MeshInstance::material_tag_mask; }
    [[nodiscard]] auto material_buffer_id() const noexcept { return material_buffer_id_and_tag >> luisa::render::MeshInstance::material_buffer_id_shift; }
    [[nodiscard]] auto light_tag() const noexcept { return light_buffer_id_and_tag & luisa::render::MeshInstance::light_tag_mask; }
    [[nodiscard]] auto light_buffer_id() const noexcept { return light_buffer_id_and_tag >> luisa::render::MeshInstance::light_buffer_id_shift; }
    [[nodiscard]] auto material_flags() const noexcept { return material_and_light_property_flags & 0xffffu; }
    [[nodiscard]] auto light_flags() const noexcept { return material_and_light_property_flags >> 16u; }
    [[nodiscard]] auto black_material() const noexcept { return (material_flags() & luisa::render::Material::property_flag_black) != 0u; }
    [[nodiscard]] auto black_light() const noexcept { return (material_flags() & luisa::render::Light::property_flag_black) != 0u; }
};

// clang-format on

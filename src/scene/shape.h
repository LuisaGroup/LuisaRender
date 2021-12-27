//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <rtx/mesh.h>
#include <scene/scene_node.h>

namespace luisa::render {

struct alignas(16) VertexAttribute {
    uint compressed_normal;
    uint compressed_tangent;
    float compressed_uv[2];

    [[nodiscard]] static constexpr auto encode(float3 normal, float3 tangent, float2 uv) noexcept {
        constexpr auto oct_encode = [](float3 n) noexcept {
            constexpr auto oct_wrap = [](float2 v) noexcept {
                return (1.0f - abs(v.yx())) * select(make_float2(-1.0f), make_float2(1.0f), v >= 0.0f);
            };
            auto p = n.xy() * (1.0f / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z)));
            p = n.z >= 0.0f ? p : oct_wrap(p);// in [-1, 1]
            auto u = make_uint2(clamp(round((p * 0.5f + 0.5f) * 65535.0f), 0.0f, 65535.0f));
            return u.x | (u.y << 16u);
        };
        return VertexAttribute{oct_encode(normal), oct_encode(tangent), uv.x, uv.y};
    };
};

struct alignas(16) Instance {

    static constexpr auto instance_buffer_id_shift = 12u;
    static constexpr auto instance_buffer_offset_mask = (1u << instance_buffer_id_shift) - 1u;

    static constexpr auto position_buffer_id_shift = 13u;
    static constexpr auto attribute_buffer_id_shift = 13u;
    static constexpr auto triangle_buffer_id_shift = 13u;
    static constexpr auto transform_buffer_id_shift = 13u;
    static constexpr auto area_cdf_buffer_id_shift = 13u;

    static constexpr auto position_buffer_element_alignment = 16u;
    static constexpr auto attribute_buffer_element_alignment = 16u;
    static constexpr auto triangle_buffer_element_alignment = 16u;
    static constexpr auto area_cdf_buffer_element_alignment = 16u;

    static constexpr auto position_buffer_offset_mask = (1u << position_buffer_id_shift) - 1u;
    static constexpr auto attribute_buffer_offset_mask = (1u << attribute_buffer_id_shift) - 1u;
    static constexpr auto triangle_buffer_offset_mask = (1u << triangle_buffer_id_shift) - 1u;
    static constexpr auto transform_buffer_offset_mask = (1u << transform_buffer_id_shift) - 1u;
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

    // transform & sampling
    uint transform_buffer_id_and_offset;// (buffer_id << shift) | offset
    uint area_cdf_buffer_id_and_offset;

    // appearance & illumination
    uint material_buffer_id_and_tag;// = (buffer_id << shift) | tag
    uint light_buffer_id_and_tag;   // = (buffer_id << shift) | tag
};

static_assert(sizeof(Instance) == 32);

class Light;
class Material;
class Transform;

class Shape : public SceneNode {

private:
    const Material *_material;
    const Light *_light;
    const Transform *_transform;

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto material() const noexcept { return _material; }
    [[nodiscard]] auto light() const noexcept { return _light; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] virtual luisa::span<float3> position_buffer() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<VertexAttribute> attribute_buffer() const noexcept = 0;
    [[nodiscard]] virtual bool is_static() const noexcept = 0;
    [[nodiscard]] virtual bool is_rigid() const noexcept = 0;
    [[nodiscard]] virtual bool is_mesh() const noexcept = 0;
    [[nodiscard]] virtual size_t child_count() const noexcept = 0;
    [[nodiscard]] virtual const Shape *child(size_t index) const noexcept = 0;
};

}// namespace luisa::render

// clang-format off

LUISA_STRUCT(
    luisa::render::VertexAttribute,
    compressed_normal,
    compressed_tangent,
    compressed_uv) {

    [[nodiscard]] static auto oct_decode(luisa::compute::Expr<uint> u) noexcept {
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
    luisa::render::Instance,

    position_buffer_id_and_offset,
    attribute_buffer_id_and_offset,
    triangle_buffer_id_and_offset,
    triangle_buffer_size,

    transform_buffer_id_and_offset,
    area_cdf_buffer_id_and_offset,
    material_buffer_id_and_tag,
    light_buffer_id_and_tag) {

    [[nodiscard]] auto position_buffer_id() const noexcept { return position_buffer_id_and_offset >> luisa::render::Instance::position_buffer_id_shift; }
    [[nodiscard]] auto position_buffer_offset() const noexcept { return (position_buffer_id_and_offset & luisa::render::Instance::position_buffer_offset_mask) * luisa::render::Instance::position_buffer_element_alignment; }
    [[nodiscard]] auto attribute_buffer_id() const noexcept { return attribute_buffer_id_and_offset >> luisa::render::Instance::attribute_buffer_id_shift; }
    [[nodiscard]] auto attribute_buffer_offset() const noexcept { return (attribute_buffer_id_and_offset & luisa::render::Instance::attribute_buffer_offset_mask) * luisa::render::Instance::attribute_buffer_element_alignment; }
    [[nodiscard]] auto triangle_buffer_id() const noexcept { return triangle_buffer_id_and_offset >> luisa::render::Instance::triangle_buffer_id_shift; }
    [[nodiscard]] auto triangle_buffer_offset() const noexcept { return (triangle_buffer_id_and_offset & luisa::render::Instance::triangle_buffer_offset_mask) * luisa::render::Instance::triangle_buffer_element_alignment; }
    [[nodiscard]] auto triangle_count() const noexcept { return triangle_buffer_size; }
    [[nodiscard]] auto transform_buffer_id() const noexcept { return transform_buffer_id_and_offset >> luisa::render::Instance::transform_buffer_id_shift; }
    [[nodiscard]] auto transform_buffer_offset() const noexcept { return transform_buffer_id_and_offset & luisa::render::Instance::transform_buffer_offset_mask; }
    [[nodiscard]] auto area_cdf_buffer_id() const noexcept { return area_cdf_buffer_id_and_offset >> luisa::render::Instance::area_cdf_buffer_id_shift; }
    [[nodiscard]] auto area_cdf_buffer_offset() const noexcept { return (area_cdf_buffer_id_and_offset & luisa::render::Instance::area_cdf_buffer_offset_mask) * luisa::render::Instance::area_cdf_buffer_element_alignment; }
    [[nodiscard]] auto material_tag() const noexcept { return material_buffer_id_and_tag & luisa::render::Instance::material_tag_mask; }
    [[nodiscard]] auto material_buffer_id() const noexcept { return material_buffer_id_and_tag >> luisa::render::Instance::material_buffer_id_shift; }
    [[nodiscard]] auto light_tag() const noexcept { return light_buffer_id_and_tag & luisa::render::Instance::light_tag_mask; }
    [[nodiscard]] auto light_buffer_id() const noexcept { return light_buffer_id_and_tag >> luisa::render::Instance::light_buffer_id_shift; }
};

// clang-format on

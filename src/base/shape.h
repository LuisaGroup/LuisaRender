//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <rtx/mesh.h>
#include <util/half.h>
#include <util/imageio.h>
#include <base/scene_node.h>

namespace luisa::render {

class Light;
class Surface;

using compute::AccelBuildHint;
using compute::Triangle;

class Light;
class Surface;
class Transform;

class Shape : public SceneNode {

public:
    class Handle;
    class VertexAttribute;

    enum struct Tag {
        MESH,
        INSTANCE,
        VIRTUAL
    };

public:
    static constexpr auto property_flag_two_sided = 1u << 0u;
    static constexpr auto property_flag_has_surface = 1u << 1u;
    static constexpr auto property_flag_has_light = 1u << 2u;

private:
    const Surface *_surface;
    const Light *_light;
    const Transform *_transform;
    luisa::optional<bool> _two_sided;

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto surface() const noexcept { return _surface; }
    [[nodiscard]] auto light() const noexcept { return _light; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] auto two_sided() const noexcept { return _two_sided; }
    [[nodiscard]] virtual bool is_mesh() const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const float3> positions() const noexcept = 0;                        // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const VertexAttribute> attributes() const noexcept = 0;              // empty if the shape is not a mesh or the mesh has no attributes
    [[nodiscard]] virtual luisa::span<const Triangle> triangles() const noexcept = 0;                      // empty if the shape is not a mesh
    [[nodiscard]] virtual luisa::span<const Shape *const> children() const noexcept = 0;                   // empty if the shape is a mesh
    [[nodiscard]] virtual bool deformable() const noexcept = 0;                                            // true if the shape will not deform
    [[nodiscard]] virtual AccelBuildHint build_hint() const noexcept { return AccelBuildHint::FAST_TRACE; }// accel struct build quality, only considered for meshes
};

struct alignas(16) Shape::VertexAttribute {

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
        return Shape::VertexAttribute{
            .compressed_normal = oct_encode(normal),
            .compressed_tangent = oct_encode(tangent),
            .compressed_uv = {uv.x, uv.y}};
    };
};

struct alignas(16) Shape::Handle {

    static constexpr auto property_flag_bits = 10u;
    static constexpr auto property_flag_mask = (1u << property_flag_bits) - 1u;

    static constexpr auto buffer_base_max = (1u << (32u - property_flag_bits)) - 1u;

    static constexpr auto light_tag_bits = 16u;
    static constexpr auto surface_tag_bits = 32u - light_tag_bits;
    static constexpr auto surface_tag_max = (1u << surface_tag_bits) - 1u;
    static constexpr auto light_tag_mask = (1u << light_tag_bits) - 1u;

    static constexpr auto position_buffer_id_offset = 0u;
    static constexpr auto attribute_buffer_id_offset = 1u;
    static constexpr auto triangle_buffer_id_offset = 2u;
    static constexpr auto alias_table_buffer_id_offset = 3u;
    static constexpr auto pdf_buffer_id_offset = 4u;

    uint buffer_base_and_properties;
    uint surface_tag_and_light_tag;
    uint triangle_buffer_size;
    uint reserved;

    [[nodiscard]] static auto encode(uint buffer_base, uint flags, uint surface_tag, uint light_tag, uint tri_count) noexcept {
        LUISA_ASSERT(buffer_base <= buffer_base_max, "Invalid geometry buffer base: {}.", buffer_base);
        LUISA_ASSERT(flags <= property_flag_mask, "Invalid property flags: {:016x}.", flags);
        LUISA_ASSERT(surface_tag <= surface_tag_max, "Invalid surface tag: {}.", surface_tag);
        LUISA_ASSERT(light_tag <= light_tag_mask, "Invalid light tag: {}.", light_tag);
        return Handle{.buffer_base_and_properties = (buffer_base << property_flag_bits) | flags,
                      .surface_tag_and_light_tag = (surface_tag << light_tag_bits) | light_tag,
                      .triangle_buffer_size = tri_count};
    }
};

static_assert(sizeof(Shape::VertexAttribute) == 16u);
static_assert(sizeof(Shape::Handle) == 16u);

}// namespace luisa::render

// clang-format off

LUISA_STRUCT(
    luisa::render::Shape::VertexAttribute,
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
    luisa::render::Shape::Handle,
    buffer_base_and_properties,
    surface_tag_and_light_tag,
    triangle_buffer_size,
    reserved) {

    [[nodiscard]] auto geometry_buffer_base() const noexcept { return buffer_base_and_properties >> luisa::render::Shape::Handle::property_flag_bits; }
    [[nodiscard]] auto property_flags() const noexcept { return buffer_base_and_properties & luisa::render::Shape::Handle::property_flag_mask; }
    [[nodiscard]] auto position_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::position_buffer_id_offset; }
    [[nodiscard]] auto attribute_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::attribute_buffer_id_offset; }
    [[nodiscard]] auto triangle_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::triangle_buffer_id_offset; }
    [[nodiscard]] auto triangle_count() const noexcept { return triangle_buffer_size; }
    [[nodiscard]] auto alias_table_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::alias_table_buffer_id_offset; }
    [[nodiscard]] auto pdf_buffer_id() const noexcept { return geometry_buffer_base() + luisa::render::Shape::Handle::pdf_buffer_id_offset; }
    [[nodiscard]] auto surface_tag() const noexcept { return surface_tag_and_light_tag >> luisa::render::Shape::Handle::light_tag_bits; }
    [[nodiscard]] auto light_tag() const noexcept { return surface_tag_and_light_tag & luisa::render::Shape::Handle::light_tag_mask; }
    [[nodiscard]] auto test_property_flag(luisa::uint flag) const noexcept { return (property_flags() & flag) != 0u; }
    [[nodiscard]] auto two_sided() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_two_sided); }
    [[nodiscard]] auto has_light() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_light); }
    [[nodiscard]] auto has_surface() const noexcept { return test_property_flag(luisa::render::Shape::property_flag_has_surface); }
};

// clang-format on

//
// Created by Mike on 2021/12/14.
//

#include <sdl/scene_node_desc.h>
#include <base/texture.h>
#include <base/surface.h>
#include <base/light.h>
#include <base/transform.h>
#include <base/scene.h>
#include <base/shape.h>

namespace luisa::render {

Shape::Shape(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SHAPE},
      _surface{scene->load_surface(desc->property_node_or_default("surface"))},
      _light{scene->load_light(desc->property_node_or_default("light"))},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))},
      _medium{scene->load_medium(desc->property_node_or_default("medium"))} {}

AccelOption Shape::build_option() const noexcept { return {}; }

bool Shape::visible() const noexcept { return true; }
float Shape::shadow_terminator_factor() const noexcept { return 0.f; }
float Shape::intersection_offset_factor() const noexcept { return 0.f; }
const Surface *Shape::surface() const noexcept { return _surface; }
const Light *Shape::light() const noexcept { return _light; }
const Medium *Shape::medium() const noexcept { return _medium; }
const Transform *Shape::transform() const noexcept { return _transform; }

bool Shape::has_vertex_normal() const noexcept {
    return is_mesh() && (vertex_properties() & property_flag_has_vertex_normal) != 0u;
}

bool Shape::has_vertex_uv() const noexcept {
    return is_mesh() && (vertex_properties() & property_flag_has_vertex_uv) != 0u;
}

bool Shape::is_mesh() const noexcept { return false; }
uint Shape::vertex_properties() const noexcept { return 0u; }
MeshView Shape::mesh() const noexcept { return {}; }
luisa::span<const Shape *const> Shape::children() const noexcept { return {}; }
bool Shape::deformable() const noexcept { return false; }

uint4 Shape::Handle::encode(
    uint buffer_base, uint flags,
    uint surface_tag, uint light_tag, uint medium_tag,
    uint tri_count, float shadow_terminator, float intersection_offset) noexcept {
    LUISA_ASSERT(buffer_base <= buffer_base_max, "Invalid geometry buffer base: {}.", buffer_base);
    LUISA_ASSERT(flags <= property_flag_mask, "Invalid property flags: {:016x}.", flags);
    LUISA_ASSERT(surface_tag <= surface_tag_max, "Invalid surface tag: {}.", surface_tag);
    LUISA_ASSERT(light_tag <= light_tag_max, "Invalid light tag: {}.", light_tag);
    LUISA_ASSERT(medium_tag <= medium_tag_max, "Invalid medium tag: {}.", medium_tag);
    constexpr auto fixed_point_bits = 16u;
    auto encode_fixed_point = [](float x) noexcept {
        x = std::clamp(x, 0.f, 1.f);
        constexpr auto fixed_point_mask = (1u << fixed_point_bits) - 1u;
        constexpr auto fixed_point_scale = 1.f / static_cast<float>(1u << fixed_point_bits);
        return static_cast<uint>(std::clamp(round(x / fixed_point_scale), 0.f, static_cast<float>(fixed_point_mask)));
    };
    auto buffer_base_and_properties = (buffer_base << property_flag_bits) | flags;
    auto tags = (surface_tag << surface_tag_offset) | (light_tag << light_tag_offset) | (medium_tag << medium_tag_offset);
    auto shadow_term_and_intersection_offset = (encode_fixed_point(shadow_terminator) << 16u) |
                                               encode_fixed_point(intersection_offset);
    return make_uint4(buffer_base_and_properties,
                      tags,
                      tri_count,
                      shadow_term_and_intersection_offset);
}

Shape::Handle Shape::Handle::decode(Expr<uint4> compressed) noexcept {
    auto buffer_base_and_properties = compressed.x;
    auto tags = compressed.y;
    auto triangle_buffer_size = compressed.z;
    auto shadow_term_and_intersection_offset = compressed.w;
    auto buffer_base = buffer_base_and_properties >> property_flag_bits;
    auto flags = buffer_base_and_properties & property_flag_mask;
    auto surface_tag = (tags >> surface_tag_offset) & surface_tag_max;
    auto light_tag = (tags >> light_tag_offset) & light_tag_max;
    auto medium_tag = (tags >> medium_tag_offset) & medium_tag_max;
    static constexpr auto decode_fixed_point = [](Expr<uint> x) noexcept {
        constexpr auto fixed_point_bits = 16u;
        constexpr auto fixed_point_mask = (1u << fixed_point_bits) - 1u;
        constexpr auto fixed_point_scale = 1.0f / static_cast<float>(1u << fixed_point_bits);
        return cast<float>(x & fixed_point_mask) * fixed_point_scale;
    };
    auto shadow_terminator = decode_fixed_point(shadow_term_and_intersection_offset >> 16u);
    auto intersection_offset = decode_fixed_point(shadow_term_and_intersection_offset & 0xffffu);
    return {buffer_base, flags, surface_tag, light_tag, medium_tag,
            triangle_buffer_size, shadow_terminator,
            clamp(intersection_offset * 255.f + 1.f, 1.f, 256.f)};
}

}// namespace luisa::render

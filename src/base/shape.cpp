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
      _transform{scene->load_transform(desc->property_node_or_default("transform"))} {}

AccelUsageHint Shape::build_hint() const noexcept { return AccelUsageHint::FAST_TRACE; }

bool Shape::visible() const noexcept { return true; }
float Shape::shadow_terminator_factor() const noexcept { return 0.f; }
float Shape::intersection_offset_factor() const noexcept { return 0.f; }
const Surface *Shape::surface() const noexcept { return _surface; }
const Light *Shape::light() const noexcept { return _light; }
const Transform *Shape::transform() const noexcept { return _transform; }

bool Shape::has_vertex_normal() const noexcept {
    return is_mesh() && (vertex_properties() & property_flag_has_vertex_normal) != 0u;
}

bool Shape::has_vertex_uv() const noexcept {
    return is_mesh() && (vertex_properties() & property_flag_has_vertex_uv) != 0u;
}

bool Shape::has_vertex_tangent() const noexcept {
    return is_mesh() && (vertex_properties() & property_flag_has_vertex_tangent) != 0u;
}

bool Shape::has_vertex_color() const noexcept {
    return is_mesh() && (vertex_properties() & property_flag_has_vertex_color) != 0u;
}

bool Shape::is_mesh() const noexcept { return false; }
uint Shape::vertex_properties() const noexcept { return 0u; }
luisa::span<const Vertex> Shape::vertices() const noexcept { return {}; }
luisa::span<const Triangle> Shape::triangles() const noexcept { return {}; }
luisa::span<const Shape *const> Shape::children() const noexcept { return {}; }
bool Shape::deformable() const noexcept { return false; }

Shape::Handle Shape::Handle::encode(uint buffer_base, uint flags, uint surface_tag, uint light_tag,
                                    uint tri_count, float shadow_terminator, float intersection_offset) noexcept {
    LUISA_ASSERT(buffer_base <= buffer_base_max, "Invalid geometry buffer base: {}.", buffer_base);
    LUISA_ASSERT(flags <= property_flag_mask, "Invalid property flags: {:016x}.", flags);
    LUISA_ASSERT(surface_tag <= surface_tag_max, "Invalid surface tag: {}.", surface_tag);
    LUISA_ASSERT(light_tag <= light_tag_mask, "Invalid light tag: {}.", light_tag);
    constexpr auto fixed_point_bits = 16u;
    auto encode_fixed_point = [](float x) noexcept {
        x = std::clamp(x, 0.f, 1.f);
        constexpr auto fixed_point_mask = (1u << fixed_point_bits) - 1u;
        constexpr auto fixed_point_scale = 1.f / static_cast<float>(1u << fixed_point_bits);
        return static_cast<uint>(std::clamp(round(x / fixed_point_scale), 0.f, static_cast<float>(fixed_point_mask)));
    };
    return Handle{.buffer_base_and_properties = (buffer_base << property_flag_bits) | flags,
                  .surface_tag_and_light_tag = (surface_tag << light_tag_bits) | light_tag,
                  .triangle_buffer_size = tri_count,
                  .shadow_term_and_intersection_offset =
                      (encode_fixed_point(shadow_terminator) << 16u) |
                      encode_fixed_point(intersection_offset)};
}

}// namespace luisa::render

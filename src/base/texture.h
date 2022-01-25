//
// Created by Mike Smith on 2022/1/25.
//

#pragma once

#include <dsl/syntax.h>
#include <base/scene_node.h>

namespace luisa::render {

struct TextureHandle {

    static constexpr auto tag_rsp_constant = 0u;      // Constant: RGB polynomial sigmoid
    static constexpr auto tag_rsp_scale_constant = 1u;// Constant: RGB polynomial sigmoid + scale
    static constexpr auto tag_srgb_texture = 2u;      // Texture: sRGB encoding
    static constexpr auto tag_gamma_texture = 3u;     // Texture: gamma encoding
    static constexpr auto tag_linear_texture = 4u;    // Texture: linear encoding
    static constexpr auto tag_rsp_texture = 5u;       // Texture: RGB polynomial sigmoid
    static constexpr auto tag_rsp_scale_texture = 6u; // Texture: RGB polynomial sigmoid + scale

    static constexpr auto texture_id_offset_shift = 6u;
    static constexpr auto encoding_tag_mask = (1 << texture_id_offset_shift) - 1u;
    static constexpr auto fixed_point_scale_max = 4096.0f;
    static constexpr auto fixed_point_scale_multiplier = 16384.0f;

    float compressed_rsp[3];
    uint texture_or_scale;

    [[nodiscard]] static TextureHandle encode_rsp_constant(float3 rsp) noexcept;
    [[nodiscard]] static TextureHandle encode_rsp_scale_constant(float3 rsp, float scale) noexcept;
    [[nodiscard]] static TextureHandle encode_srgb_texture(uint tex_id) noexcept;
    [[nodiscard]] static TextureHandle encode_gamma_texture(uint tex_id) noexcept;
    [[nodiscard]] static TextureHandle encode_linear_texture(uint tex_id) noexcept;
    [[nodiscard]] static TextureHandle encode_rsp_texture(uint tex_id) noexcept;
    [[nodiscard]] static TextureHandle encode_rsp_scale_texture(uint tex_id) noexcept;
    [[nodiscard]] static TextureHandle encode_custom(uint custom_tag, float3 custom_float3, uint custom_id) noexcept;
};

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::TextureHandle, compressed_rsp, texture_or_scale) {
    [[nodiscard]] auto rsp() const noexcept {
        return luisa::compute::def<luisa::float3>(compressed_rsp);
    }
    [[nodiscard]] auto tag() const noexcept {
        return texture_or_scale & luisa::render::TextureHandle::encoding_tag_mask;
    }
    [[nodiscard]] auto scale() const noexcept {
        using luisa::compute::cast;
        using luisa::render::TextureHandle;
        return cast<float>(texture_or_scale >> TextureHandle::texture_id_offset_shift) *
            (1.0f / TextureHandle::fixed_point_scale_multiplier);
    }
    [[nodiscard]] auto texture_id() const noexcept {
        return texture_or_scale >> luisa::render::TextureHandle::texture_id_offset_shift;
    }
    [[nodiscard]] auto custom_tag() const noexcept { return tag(); }
    [[nodiscard]] auto custom_float3() const noexcept { return rsp(); }
    [[nodiscard]] auto custom_id() const noexcept { return texture_id(); }
};
// clang-format on

namespace luisa::render {

class Texture : public SceneNode {

public:
    Texture(Scene *scene, const SceneNodeDesc *desc) noexcept;

};

}

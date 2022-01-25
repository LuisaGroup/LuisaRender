//
// Created by Mike Smith on 2022/1/25.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

TextureHandle render::TextureHandle::encode_rsp_constant(float3 rsp) noexcept {
    return TextureHandle{
        .compressed_rsp = {rsp.x, rsp.y, rsp.z},
        .texture_or_scale = tag_rsp_constant};
}

TextureHandle render::TextureHandle::encode_rsp_scale_constant(float3 rsp, float scale) noexcept {
    auto fp_scale = static_cast<uint>(std::round(
        std::clamp(scale, 0.0f, fixed_point_scale_max) *
        fixed_point_scale_multiplier));
    return TextureHandle{
        .compressed_rsp = {rsp.x, rsp.y, rsp.z},
        .texture_or_scale = tag_rsp_scale_constant | (fp_scale << texture_id_offset_shift)};
}

[[nodiscard]] static inline auto texture_handle_encode_texture(uint tag, uint tex_id) noexcept {
    if (tex_id > (~0u >> TextureHandle::texture_id_offset_shift)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture id for texture handle: {}.",
            tex_id);
    }
    constexpr auto shift = TextureHandle::texture_id_offset_shift;
    return TextureHandle{.compressed_rsp = {}, .texture_or_scale = tag | (tex_id << shift)};
}

TextureHandle render::TextureHandle::encode_srgb_texture(uint tex_id) noexcept {
    return texture_handle_encode_texture(tag_srgb_texture, tex_id);
}

TextureHandle render::TextureHandle::encode_gamma_texture(uint tex_id) noexcept {
    return texture_handle_encode_texture(tag_gamma_texture, tex_id);
}

TextureHandle render::TextureHandle::encode_linear_texture(uint tex_id) noexcept {
    return texture_handle_encode_texture(tag_linear_texture, tex_id);
}

TextureHandle render::TextureHandle::encode_rsp_texture(uint tex_id) noexcept {
    return texture_handle_encode_texture(tag_rsp_texture, tex_id);
}

TextureHandle render::TextureHandle::encode_rsp_scale_texture(uint tex_id) noexcept {
    return texture_handle_encode_texture(tag_rsp_scale_texture, tex_id);
}

TextureHandle TextureHandle::encode_custom(uint custom_tag, float3 custom_float3, uint custom_id) noexcept {
    if (custom_tag < tag_custom_begin || custom_tag > tag_mask) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid custom tag for texture handle: {}.",
            custom_tag);
    }
    if (custom_id > (~0u >> texture_id_offset_shift)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid custom id for texture handle: {}.",
            custom_id);
    }
    return TextureHandle{
        .compressed_rsp = {custom_float3.x, custom_float3.y, custom_float3.z},
        .texture_or_scale = custom_tag | (custom_id << texture_id_offset_shift)};
}

Texture::Texture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE} {}

uint Texture::handle_tag() const noexcept {
    static std::mutex mutex;
    static luisa::unordered_map<luisa::string, uint, Hash64> impl_to_tag;
    std::scoped_lock lock{mutex};
    auto [iter, _] = impl_to_tag.try_emplace(
        luisa::string{impl_type()},
        static_cast<uint>(impl_to_tag.size() + TextureHandle::tag_custom_begin));
    return iter->second;
}

TextureHandle Texture::encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept {
    auto handle = _encode(pipeline, command_buffer, instance_id, shape);
    pipeline.register_texture(this);
    return handle;
}

}// namespace luisa::render

//
// Created by Mike Smith on 2022/1/25.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

Texture::Texture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE} {}

TextureHandle TextureHandle::encode_constant(uint tag, float3 v, float alpha) noexcept {
    if (tag > tag_mask) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid tag for texture handle: {}.", tag);
    }
    return {.id_and_tag = tag | (float_to_half(alpha) << texture_id_offset_shift),
            .compressed_v = {v.x, v.y, v.z}};
}

TextureHandle TextureHandle::encode_texture(uint tag, uint tex_id, float3 v) noexcept {
    if (tag > tag_mask) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid tag for texture handle: {}.", tag);
    }
    if (tex_id > (~0u >> texture_id_offset_shift)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid id for texture handle: {}.", tex_id);
    }
    return {.id_and_tag = tag | (tex_id << texture_id_offset_shift),
            .compressed_v = {v.x, v.y, v.z}};
}

ImageTexture::ImageTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : Texture{scene, desc} {
    auto filter = desc->property_string_or_default("filter", "bilinear");
    auto address = desc->property_string_or_default("address", "repeat");
    for (auto &c : filter) { c = static_cast<char>(tolower(c)); }
    for (auto &c : address) { c = static_cast<char>(tolower(c)); }
    auto address_mode = [&address, desc] {
        for (auto &c : address) { c = static_cast<char>(tolower(c)); }
        if (address == "zero") { return TextureSampler::Address::ZERO; }
        if (address == "edge") { return TextureSampler::Address::EDGE; }
        if (address == "mirror") { return TextureSampler::Address::MIRROR; }
        if (address == "repeat") { return TextureSampler::Address::REPEAT; }
        LUISA_ERROR(
            "Invalid texture address mode '{}'. [{}]",
            address, desc->source_location().string());
    }();
    auto filter_mode = [&filter, desc] {
        for (auto &c : filter) { c = static_cast<char>(tolower(c)); }
        if (filter == "point") { return TextureSampler::Filter::POINT; }
        if (filter == "bilinear") { return TextureSampler::Filter::LINEAR_POINT; }
        if (filter == "trilinear") { return TextureSampler::Filter::LINEAR_LINEAR; }
        if (filter == "anisotropic" || filter == "aniso") { return TextureSampler::Filter::ANISOTROPIC; }
        LUISA_ERROR(
            "Invalid texture filter mode '{}'. [{}]",
            filter, desc->source_location().string());
    }();
    _sampler = {filter_mode, address_mode};
    _uv_scale = desc->property_float2_or_default(
        "uv_scale", lazy_construct([desc] {
            return make_float2(desc->property_float_or_default(
                "uv_scale", 1.0f));
        }));
    _uv_offset = desc->property_float2_or_default(
        "uv_offset", lazy_construct([desc] {
            return make_float2(desc->property_float_or_default(
                "uv_offset", 0.0f));
        }));
}

Float4 ImageTexture::evaluate(
    const Pipeline &pipeline, const Interaction &it,
    const Var<TextureHandle> &handle, Expr<float>) const noexcept {
    auto uv_scale = as<uint>(handle->v().x);
    auto u_scale = half_to_float(uv_scale & 0xffffu);
    auto v_scale = half_to_float(uv_scale >> 16u);
    auto uv_offset = handle->v().yz();
    auto uv = it.uv() * make_float2(u_scale, v_scale) + uv_offset;
    return pipeline.tex2d(handle->texture_id()).sample(uv);// TODO: LOD
}

TextureHandle ImageTexture::_encode(
    Pipeline &pipeline, CommandBuffer &command_buffer,
    uint handle_tag) const noexcept {

    auto &&image = _image();
    auto device_image = pipeline.create<Image<float>>(image.pixel_storage(), image.size());
    auto tex_id = pipeline.register_bindless(*device_image, _sampler);
    command_buffer << device_image->copy_from(image.pixels())
                   << compute::commit();
    auto u_scale = float_to_half(_uv_scale.x);
    auto v_scale = float_to_half(_uv_scale.y);
    auto compressed = make_float3(
        luisa::bit_cast<float>(u_scale | (v_scale << 16u)),
                _uv_offset);
    return TextureHandle::encode_texture(
        handle_tag, tex_id, compressed);
}

}// namespace luisa::render

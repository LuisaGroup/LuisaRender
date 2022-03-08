//
// Created by Mike Smith on 2022/1/25.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

Texture::Texture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE} {}

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

luisa::unique_ptr<Texture::Instance> ImageTexture::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto &&image = _image();
    auto device_image = pipeline.create<Image<float>>(image.pixel_storage(), image.size());
    auto tex_id = pipeline.register_bindless(*device_image, _sampler);
    command_buffer << device_image->copy_from(image.pixels())
                   << compute::commit();
    return luisa::make_unique<Instance>(pipeline, this, tex_id);
}

Float4 ImageTexture::Instance::evaluate(const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto texture = node<ImageTexture>();
    auto uv_scale = texture->uv_scale();
    auto uv_offset = texture->uv_offset();
    auto uv = it.uv() * uv_scale + uv_offset;
    auto v = pipeline().tex2d(_texture_id).sample(uv);// TODO: LOD
    switch (texture->category()) {
        case Category::COLOR: {
            auto rsp = RGBSigmoidPolynomial{v.xyz()};
            auto spec = RGBAlbedoSpectrum{rsp};
            return spec.sample(swl);
        }
        case Category::ILLUMINANT: {
            auto rsp = RGBSigmoidPolynomial{v.xyz()};
            auto spec = RGBIlluminantSpectrum{
                rsp, v.w, DenselySampledSpectrum::cie_illum_d65()};
            return spec.sample(swl);
        }
        default: break;
    }
    return v;
}

}// namespace luisa::render

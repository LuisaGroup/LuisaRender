//
// Created by Mike Smith on 2022/1/25.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

Texture::Texture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE} {}

luisa::optional<float4> Texture::evaluate_static() const noexcept { return luisa::nullopt; }

[[nodiscard]] inline auto extend_color_to_rgb(auto color, uint n) noexcept {
    if (n == 1u) { return color.xxx(); }
    if (n == 2u) { return make_float3(color.xy(), 1.f); }
    return color;
}

Spectrum::Decode Texture::Instance::evaluate_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    // skip the expensive encoding/decoding if the texture is static
    if (auto v = node()->evaluate_static()) {
        return _evaluate_static_albedo_spectrum(swl, *v);
    }
    // we have got no luck, do the expensive encoding/decoding
    auto v = evaluate(it, swl, time);
    v = pipeline().spectrum()->encode_srgb_albedo(
        extend_color_to_rgb(v.xyz(), node()->channels()));
    return pipeline().spectrum()->decode_albedo(swl, v);
}

Spectrum::Decode Texture::Instance::evaluate_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    // skip the expensive encoding/decoding if the texture is static
    if (auto v = node()->evaluate_static()) {
        return _evaluate_static_illuminant_spectrum(swl, *v);
    }
    // we have got no luck, do the expensive encoding/decoding
    auto v = evaluate(it, swl, time);
    v = pipeline().spectrum()->encode_srgb_illuminant(v.xyz());
    return pipeline().spectrum()->decode_illuminant(swl, v);
}

Spectrum::Decode Texture::Instance::_evaluate_static_albedo_spectrum(
    const SampledWavelengths &swl, float4 v) const noexcept {
    auto enc = pipeline().spectrum()->node()->encode_static_srgb_albedo(
        extend_color_to_rgb(v.xyz(), node()->channels()));
    return pipeline().spectrum()->decode_albedo(swl, enc);
}

Spectrum::Decode Texture::Instance::_evaluate_static_illuminant_spectrum(
    const SampledWavelengths &swl, float4 v) const noexcept {
    auto enc = pipeline().spectrum()->node()->encode_static_srgb_illuminant(
        extend_color_to_rgb(v.xyz(), node()->channels()));
    return pipeline().spectrum()->decode_illuminant(swl, enc);
}

}// namespace luisa::render

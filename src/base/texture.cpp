//
// Created by Mike Smith on 2022/1/25.
//

#include "util/spec.h"
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

Texture::Texture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE},
      _range{desc->property_float2_or_default(
          "range", make_float2(std::numeric_limits<float>::min(),
                               std::numeric_limits<float>::max()))},
      _requires_grad{desc->property_bool_or_default("requires_grad", false)},
      _render_grad_map{desc->property_bool_or_default("render_grad_map", false)} {}

bool Texture::requires_gradients() const noexcept { return _requires_grad; }
bool Texture::render_grad_map() const noexcept { return _render_grad_map; }
void Texture::disable_gradients() noexcept { _requires_grad = false; }

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

Spectrum::Decode Texture::Instance::evaluate_unbounded_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    // skip the expensive encoding/decoding if the texture is static
    if (auto v = node()->evaluate_static()) {
        return _evaluate_static_unbounded_spectrum(swl, *v);
    }
    // we have got no luck, do the expensive encoding/decoding
    auto v = evaluate(it, swl, time);
    v = pipeline().spectrum()->encode_srgb_unbounded(
        extend_color_to_rgb(v.xyz(), node()->channels()));
    return pipeline().spectrum()->decode_unbounded(swl, v);
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

Spectrum::Decode Texture::Instance::_evaluate_static_unbounded_spectrum(
    const SampledWavelengths &swl, float4 v) const noexcept {
    auto enc = pipeline().spectrum()->node()->encode_static_srgb_unbounded(
        extend_color_to_rgb(v.xyz(), node()->channels()));
    return pipeline().spectrum()->decode_unbounded(swl, enc);
}

Spectrum::Decode Texture::Instance::_evaluate_static_illuminant_spectrum(
    const SampledWavelengths &swl, float4 v) const noexcept {
    auto enc = pipeline().spectrum()->node()->encode_static_srgb_illuminant(
        extend_color_to_rgb(v.xyz(), node()->channels()));
    return pipeline().spectrum()->decode_illuminant(swl, enc);
}

void Texture::Instance::backward_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    auto dEnc = pipeline().spectrum()->backward_decode_albedo(swl, evaluate(it, swl, time), dSpec);
    dEnc = make_float4(pipeline().spectrum()->backward_encode_srgb_albedo(dEnc), 1.f);
    // device_log("grad in texture_albedo: ({}, {}, {})", dEnc[0u], dEnc[1u], dEnc[2u]);
    backward(it, swl, time, dEnc);
}

SampledSpectrum Texture::Instance::eval_grad_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    auto dEnc = pipeline().spectrum()->backward_decode_albedo(swl, evaluate(it, swl, time), dSpec);
    dEnc = make_float4(pipeline().spectrum()->backward_encode_srgb_albedo(dEnc), 1.f);
    return eval_grad(it, swl, time, dEnc);
}

void Texture::Instance::backward_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    auto dEnc = pipeline().spectrum()->backward_decode_illuminant(swl, evaluate(it, swl, time), dSpec);
    dEnc = make_float4(pipeline().spectrum()->backward_encode_srgb_illuminant(dEnc), 1.f);
    backward(it, swl, time, dEnc);
}

void Texture::Instance::backward_unbounded_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    auto dEnc = pipeline().spectrum()->backward_decode_unbounded(swl, evaluate(it, swl, time), dSpec);
    dEnc = make_float4(pipeline().spectrum()->backward_encode_srgb_unbounded(dEnc), 1.f);
    backward(it, swl, time, dEnc);
}

}// namespace luisa::render

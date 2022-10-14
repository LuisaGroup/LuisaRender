//
// Created by Mike Smith on 2022/1/25.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

Texture::Texture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE},
      _range{desc->property_float2_or_default(
          "range", make_float2(std::numeric_limits<float>::min(),
                               std::numeric_limits<float>::max()))},
      _semantic{[desc] {
          auto s = desc->property_string_or_default("semantic", "generic");
          for (auto &c : s) { c = static_cast<char>(c); }
          if (s == "albedo" || s == "color") { return Semantic::ALBEDO; }
          if (s == "illuminant" || s == "illum" || s == "emission") { return Semantic::ILLUMINANT; }
          if (s == "generic") { return Semantic::GENERIC; }
          LUISA_WARNING_WITH_LOCATION("Unknown texture semantic '{}'. Fallback to generic.", s);
          return Semantic::GENERIC;
      }()},
      _requires_grad{desc->property_bool_or_default("requires_grad", false)} {}

bool Texture::requires_gradients() const noexcept { return _requires_grad; }
void Texture::disable_gradients() noexcept { _requires_grad = false; }

Spectrum::Decode Texture::Instance::evaluate_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    LUISA_ASSERT(node()->semantic() == Semantic::ALBEDO ||
                     node()->semantic() == Semantic::GENERIC,
                 "Decoding albedo spectrum with non-albedo texture.");
    auto v = evaluate(it, swl, time);
    if (_pipeline.spectrum()->node()->requires_encoding() &&
        !node()->is_spectral_encoding()) {
        v = pipeline().spectrum()->encode_srgb_albedo(v.xyz());
    }
    return pipeline().spectrum()->decode_albedo(swl, v);
}

Spectrum::Decode Texture::Instance::evaluate_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    LUISA_ASSERT(node()->semantic() == Semantic::ILLUMINANT ||
                     node()->semantic() == Semantic::GENERIC,
                 "Decoding illuminant spectrum with non-illuminant texture.");
    auto v = evaluate(it, swl, time);
    if (_pipeline.spectrum()->node()->requires_encoding() &&
        !node()->is_spectral_encoding()) {
        v = pipeline().spectrum()->encode_srgb_illuminant(v.xyz());
    }
    return pipeline().spectrum()->decode_illuminant(swl, v);
}

void Texture::Instance::backward_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    LUISA_ASSERT(node()->semantic() == Semantic::ALBEDO ||
                     node()->semantic() == Semantic::GENERIC,
                 "Decoding albedo spectrum with non-albedo texture.");
    auto dEnc = pipeline().spectrum()->backward_decode_albedo(swl, evaluate(it, swl, time), dSpec);
    if (_pipeline.spectrum()->node()->requires_encoding() &&
        !node()->is_spectral_encoding()) {
        dEnc = make_float4(pipeline().spectrum()->backward_encode_srgb_albedo(dEnc), 1.f);
    }
    backward(it, swl, time, dEnc);
}

void Texture::Instance::backward_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    LUISA_ASSERT(node()->semantic() == Semantic::ILLUMINANT ||
                     node()->semantic() == Semantic::GENERIC,
                 "Decoding illuminant spectrum with non-illuminant texture.");
    auto dEnc = pipeline().spectrum()->backward_decode_illuminant(swl, evaluate(it, swl, time), dSpec);
    if (_pipeline.spectrum()->node()->requires_encoding() &&
        !node()->is_spectral_encoding()) {
        dEnc = make_float4(pipeline().spectrum()->backward_encode_srgb_illuminant(dEnc), 1.f);
    }
    backward(it, swl, time, dEnc);
}

}// namespace luisa::render

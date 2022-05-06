//
// Created by Mike Smith on 2022/1/25.
//

#include <base/texture.h>
#include <base/pipeline.h>
#include <util/atomic.h>

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

Spectrum::Decode Texture::Instance::evaluate_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return pipeline().spectrum()->decode_albedo(swl, evaluate(it, time));
}

Spectrum::Decode Texture::Instance::evaluate_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return pipeline().spectrum()->decode_illuminant(swl, evaluate(it, time));
}

void Texture::Instance::backward_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    auto dEnc = pipeline().spectrum()->backward_decode_albedo(swl, evaluate(it, time), dSpec);
    backward(it, time, dEnc);
}

void Texture::Instance::backward_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    auto dEnc = pipeline().spectrum()->backward_decode_illuminant(swl, evaluate(it, time), dSpec);
    backward(it, time, dEnc);
}

}// namespace luisa::render

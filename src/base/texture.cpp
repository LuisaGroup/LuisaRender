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
      _requires_grad{desc->property_bool_or_default("requires_grad", false)} {}

SampledSpectrum Texture::Instance::evaluate_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return swl.albedo_from_srgb(evaluate(it, time).xyz());
}

SampledSpectrum Texture::Instance::evaluate_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return swl.illuminant_from_srgb(evaluate(it, time).xyz());
}

void Texture::Instance::backward_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    auto dRGB = swl.backward_albedo_from_srgb(evaluate(it, time).xyz(), dSpec);
    backward(it, time, make_float4(dRGB, 0.f));
}

void Texture::Instance::backward_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> time, const SampledSpectrum &dSpec) const noexcept {
    auto dRGB = swl.backward_illuminant_from_srgb(evaluate(it, time).xyz(), dSpec);
    backward(it, time, make_float4(dRGB, 0.f));
}

}// namespace luisa::render

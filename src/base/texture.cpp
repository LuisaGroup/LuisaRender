//
// Created by Mike Smith on 2022/1/25.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

Texture::Texture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE} {}

Spectrum::Decode Texture::Instance::evaluate_albedo_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto v = evaluate(it, swl, time);
    v = pipeline().spectrum()->encode_srgb_albedo(v.xyz());
    return pipeline().spectrum()->decode_albedo(swl, v);
}

Spectrum::Decode Texture::Instance::evaluate_illuminant_spectrum(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto v = evaluate(it, swl, time);
    v = pipeline().spectrum()->encode_srgb_illuminant(v.xyz());
    return pipeline().spectrum()->decode_illuminant(swl, v);
}

}// namespace luisa::render

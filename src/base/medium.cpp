//
// Created by ChenXin on 2023/2/13.
//

#include "medium.h"

namespace luisa::render {

using compute::Ray;

Medium::Medium(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::MEDIUM},
      _priority{desc->property_uint_or_default("priority", 0u)} {}

unique_ptr<Medium::Instance> Medium::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto instance = _build(pipeline, command_buffer);
    return instance;
}

Medium::Closure::Closure(
    const Medium::Instance *instance, Expr<Ray> ray, luisa::shared_ptr<Interaction> it,
    const SampledWavelengths &swl, Expr<float> time, Expr<float> eta) noexcept
    : _instance{instance}, _ray{ray}, _it{std::move(it)}, _swl{swl}, _time{time}, _eta{eta} {}

Medium::Sample Medium::Closure::sample(Expr<float> t_max, Sampler::Instance *sampler) const noexcept {
    return _sample(t_max, sampler);
}

SampledSpectrum Medium::Closure::transmittance(Expr<float> t, Sampler::Instance *sampler) const noexcept {
    return _transmittance(t, sampler);
}

}
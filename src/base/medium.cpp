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
    const Medium::Instance *instance, Expr<Ray> ray,
    const SampledWavelengths &swl, Expr<float> time, Expr<float> eta,
    const SampledSpectrum& sigma_a, const SampledSpectrum& sigma_s, const SampledSpectrum& le,
    const PhaseFunction::Instance *phase_function) noexcept
    : _instance{instance}, _ray{ray}, _swl{swl}, _time{time}, _eta{eta},
      _sigma_a{sigma_a}, _sigma_s{sigma_s}, _le{le}, _phase_function{phase_function} {}

}// namespace luisa::render
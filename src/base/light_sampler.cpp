//
// Created by Mike Smith on 2022/1/10.
//

#include <base/light_sampler.h>

namespace luisa::render {

LightSampler::LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LIGHT_SAMPLER} {}

Light::Sample LightSampler::Instance::sample(
    const Interaction &it_from, Expr<float> u_sel, Expr<float2> u_light,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto sample = Light::Sample::zero(swl.dimension());
    if (!_pipeline.has_lighting()) { return sample; }
    auto sel = select(it_from, u_sel, swl, time);
    if (_pipeline.environment() != nullptr) {// possibly environment lighting
        if (_pipeline.lights().empty()) {// no lights, just environment lighting
            sample = sample_environment(it_from.p(), u_light, swl, time);
        } else {
            $if(sel.tag == selection_environment) {
                sample = sample_environment(it_from.p(), u_light, swl, time);
            }
            $else {
                sample = sample_light(it_from, sel.tag, u_light, swl, time);
            };
        }
    } else {// no environment light
        sample = sample_light(it_from, sel.tag, u_light, swl, time);
    }
    sample.eval.pdf *= sel.prob;
    return sample;
}

}// namespace luisa::render

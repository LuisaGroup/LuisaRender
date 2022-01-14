//
// Created by Mike Smith on 2022/1/10.
//

#include <scene/light_sampler.h>

namespace luisa::render {

LightSampler::LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LIGHT_SAMPLER} {}

Light::Evaluation LightSampler::Instance::evaluate(const Interaction &it, Expr<float3> p_from, Expr<float> time) const noexcept {
    Light::Evaluation eval;
    _pipeline.decode_light(it.shape()->light_tag(), [&](const Light::Closure &light) noexcept {
        eval = light.evaluate(it, p_from, time);
    });
    eval.pdf *= pmf(it);
    return eval;
}

Light::Sample LightSampler::Instance::sample(Sampler::Instance &sampler, const Interaction &it_from, Expr<float> time) const noexcept {
    Light::Sample light_sample;
    auto selection = select(sampler, it_from);
    _pipeline.decode_light(selection.light_tag, [&](const Light::Closure &light) noexcept {
        light_sample = light.sample(sampler, selection.instance_id, it_from, time);
    });
    light_sample.eval.pdf *= selection.pmf;
    return light_sample;
}

uint LightSampler::Instance::light_count() const noexcept {
    return _pipeline.environment() == nullptr ?
               static_cast<uint>(_pipeline.lights().size()) :
               static_cast<uint>(_pipeline.lights().size() + 1u);
}

}// namespace luisa::render

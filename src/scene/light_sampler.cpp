//
// Created by Mike Smith on 2022/1/10.
//

#include <scene/light_sampler.h>

namespace luisa::render {

LightSampler::LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LIGHT_SAMPLER} {}

Light::Evaluation LightSampler::Instance::evaluate(const Interaction &it, Expr<float3> p_from) const noexcept {
    Light::Evaluation eval;
    _pipeline.decode_light(it.shape()->light_tag(), it, [&](const Light::Closure &light) noexcept {
        eval = light.evaluate(p_from);
    });
    eval.pdf *= pdf_selection(it);
    return eval;
}

Light::Sample LightSampler::Instance::sample(Sampler::Instance &sampler, const Interaction &it) const noexcept {
    Light::Sample light_sample;
    auto selection = select(sampler, it);
    _pipeline.decode_light(selection.light_tag, it, [&](const Light::Closure &light) noexcept {
        light_sample = light.sample(sampler, selection.instance_id);
    });
    light_sample.eval.pdf *= selection.pdf;
    return light_sample;
}

}

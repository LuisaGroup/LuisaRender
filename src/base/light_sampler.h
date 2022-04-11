//
// Created by Mike Smith on 2022/1/10.
//

#pragma once

#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/light.h>
#include <base/environment.h>

namespace luisa::render {

class Interaction;

using compute::Float;
using compute::UInt;

// TODO: consider environments
class LightSampler : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const LightSampler *_sampler;

    public:
        explicit Instance(const Pipeline &pipeline, const LightSampler *light_dist) noexcept
            : _pipeline{pipeline}, _sampler{light_dist} {}
        virtual ~Instance() noexcept = default;

        template<typename T = LightSampler>
            requires std::is_base_of_v<LightSampler, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_sampler); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Light::Evaluation evaluate_hit(
            const Interaction &it, Expr<float3> p_from,
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Light::Evaluation evaluate_miss(
            Expr<float3> wi, const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual Light::Sample sample(
            Sampler::Instance &sampler, const Interaction &it_from,
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        //        virtual void backward_hit(
        //            const Interaction &it, Expr<float3> p_from,
        //            const SampledWavelengths &swl, Expr<float> time,
        //            const SampledSpectrum &df) const noexcept = 0;
        //        virtual void backward_miss(
        //            Expr<float3> wi, Expr<float3x3> env_to_world,
        //            const SampledWavelengths &swl, Expr<float> time,
        //            const SampledSpectrum &df) const noexcept = 0;
    };

public:
    LightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

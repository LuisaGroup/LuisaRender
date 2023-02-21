//
// Created by ChenXin on 2023/2/13.
//

#pragma once

#include <luisa-compute.h>
#include <util/spec.h>
#include <base/scene_node.h>
#include <base/spectrum.h>
#include <base/interaction.h>

#include <utility>

namespace luisa::render {

using compute::Expr;
using compute::Var;

#define RAY_EPSILON 1e-3f

class Medium : public SceneNode {

public:
    static uint constexpr INVALID_TAG = uint(0u) - uint(1u);

protected:
    uint _priority;

public:
    struct Evaluation {
        SampledSpectrum f;

        [[nodiscard]] static auto zero(uint spec_dim) noexcept {
            return Evaluation{
                .f = SampledSpectrum{spec_dim}};
        }
    };

    struct Sample {
        Evaluation eval;
        Var<Ray> ray;
        Bool is_scattered;

        [[nodiscard]] static auto zero(uint spec_dim) noexcept {
            return Sample{.eval = Evaluation::zero(spec_dim),
                          .ray = def<Ray>(),
                          .is_scattered = false};
        }
    };

    class Instance;

    class Closure {

    private:
        const Instance *_instance;

    private:
        const SampledWavelengths &_swl;
        Var<Ray> _ray;
        luisa::shared_ptr<Interaction> _it;
        Float _time;
        Float _eta;

    protected:
        [[nodiscard]] SampledSpectrum analyticTransmittance(
            Expr<float> t,
            const SampledSpectrum &sigma) const noexcept {
            return exp(-sigma * t);
        }

    private:
        [[nodiscard]] virtual Sample _sample(Expr<float> t_max, Sampler::Instance *sampler) const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum _transmittance(Expr<float> t, Sampler::Instance *sampler) const noexcept = 0;

    public:
        Closure(const Instance *instance, Expr<Ray> ray, luisa::shared_ptr<Interaction> it,
                const SampledWavelengths &swl, Expr<float> time, Expr<float> eta) noexcept;
        virtual ~Closure() noexcept = default;
        template<typename T = Instance>
            requires std::is_base_of_v<Instance, T>
        [[nodiscard]] auto instance() const noexcept { return static_cast<const T *>(_instance); }
        [[nodiscard]] auto it() const noexcept { return _it.get(); }
        [[nodiscard]] auto shared_it() const noexcept { return _it; }
        [[nodiscard]] auto &swl() const noexcept { return _swl; }
        [[nodiscard]] auto ray() const noexcept { return _ray; }
        [[nodiscard]] auto time() const noexcept { return _time; }
        [[nodiscard]] auto eta() const noexcept { return _eta; }
        [[nodiscard]] Sample sample(Expr<float> t_max, Sampler::Instance *sampler) const noexcept;
        [[nodiscard]] SampledSpectrum transmittance(Expr<float> t, Sampler::Instance *sampler) const noexcept;
    };

    class Instance {
    protected:
        const Pipeline &_pipeline;
        const Medium *_medium;
        friend class Medium;

    public:
        [[nodiscard]] auto priority() const noexcept { return _medium->_priority; }

    public:
        Instance(const Pipeline &pipeline, const Medium *medium) noexcept
            : _pipeline{pipeline}, _medium{medium} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Medium>
            requires std::is_base_of_v<Medium, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_medium); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual luisa::unique_ptr<Closure> closure(
            Expr<Ray> ray, luisa::shared_ptr<Interaction> interaction, const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
    };

protected:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

public:
    Medium(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

}// namespace luisa::render
//
// Created by ChenXin on 2023/2/14.
//

#pragma once

#include <util/spec.h>
#include <base/scene_node.h>
#include <base/spectrum.h>
#include <util/sampling.h>

#include <utility>

namespace luisa::render {

using compute::Expr;
using compute::Var;

class PhaseFunction : public SceneNode {
public:
    struct PhaseFunctionSample {
        Float p;
        Float3 wi;
        Float pdf;
        Bool valid;
    };

    class Instance;

    class Instance {
    public:

    protected:
        const Pipeline &_pipeline;
        const PhaseFunction *_phase_function;
        friend class PhaseFunction;

    public:
        [[nodiscard]] virtual Float p(Expr<float3> wo, Expr<float3> wi) const = 0;
        [[nodiscard]] virtual PhaseFunctionSample sample_p(Expr<float3> wo, Expr<float2> u) const = 0;
        [[nodiscard]] virtual Float pdf(Expr<float3> wo, Expr<float3> wi) const = 0;

    public:
        Instance(const Pipeline &pipeline, const PhaseFunction *phase_function) noexcept
            : _pipeline{pipeline}, _phase_function{phase_function} {}
        virtual ~Instance() noexcept = default;
        template<typename T = PhaseFunction>
            requires std::is_base_of_v<PhaseFunction, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_phase_function); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
    };

protected:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

public:
    PhaseFunction(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

}// namespace luisa::render

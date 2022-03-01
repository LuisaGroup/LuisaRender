//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <rtx/ray.h>
#include <runtime/bindless_array.h>
#include <util/spectrum.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/shape.h>

namespace luisa::render {

using compute::BindlessArray;
using compute::Ray;

class Shape;
class Interaction;

class Light : public SceneNode {

public:
    struct Evaluation {
        Float4 L;
        Float pdf;
    };

    struct Sample {
        Evaluation eval;
        Var<Ray> shadow_ray;
    };

    struct Closure {
        virtual ~Closure() noexcept = default;
        [[nodiscard]] virtual Evaluation evaluate(
            const Interaction &it_light, Expr<float3> p_from) const noexcept = 0;
        [[nodiscard]] virtual Sample sample(
            Sampler::Instance &sampler, Expr<uint> light_inst_id,
            const Interaction &it_from) const noexcept = 0;
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Light *_light;

    public:
        Instance(const Pipeline &pipeline, const Light *light) noexcept
            : _pipeline{pipeline}, _light{light} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Light>
            requires std::is_base_of_v<Light, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_light); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual luisa::unique_ptr<Closure> closure(
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
    };

public:
    Light(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] virtual bool is_virtual() const noexcept { return false; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <rtx/ray.h>
#include <runtime/bindless_array.h>
#include <util/spec.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/shape.h>
#include <base/spectrum.h>

namespace luisa::render {

using compute::BindlessArray;
using compute::Ray;

class Shape;
class Interaction;

class Light : public SceneNode {

public:
    struct Handle {
        uint instance_id;
        uint light_tag;
    };

    struct Evaluation {
        SampledSpectrum L;
        Float pdf;
        [[nodiscard]] static auto zero(size_t spec_dim) noexcept {
            return Evaluation{.L = SampledSpectrum{spec_dim}, .pdf = 0.f};
        }
    };

    struct Sample {
        Evaluation eval;
        Float3 wi;
        Float distance;
        [[nodiscard]] static auto zero(size_t spec_dim) noexcept {
            return Sample{.eval = Evaluation::zero(spec_dim),
                          .wi = make_float3(0.f, 0.f, 1.f),
                          .distance = 0.f};
        }
    };

    class Instance;

    class Closure {

    private:
        const Instance *_instance;

    protected:
        const SampledWavelengths &_swl;
        Float _time;

    public:
        Closure(const Instance *instance, const SampledWavelengths &swl, Expr<float> time) noexcept
            : _instance{instance}, _swl{swl}, _time{time} {}
        virtual ~Closure() noexcept = default;
        template<typename T = Instance>
            requires std::is_base_of_v<Instance, T>
        [[nodiscard]] auto instance() const noexcept { return static_cast<const T *>(_instance); }
        [[nodiscard]] virtual Evaluation evaluate(
            const Interaction &it_light, Expr<float3> p_from) const noexcept = 0;
        [[nodiscard]] virtual Sample sample(
            Expr<uint> light_inst_id, Expr<float3> p_from,
            Expr<float> u_prim, Expr<float2> u) const noexcept = 0;
        virtual void backward(const Interaction &it_light, Expr<float3> p_from, const SampledSpectrum &df) const noexcept = 0;
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

LUISA_STRUCT(luisa::render::Light::Handle, instance_id, light_tag){};

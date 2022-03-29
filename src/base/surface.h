//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/spec.h>
#include <base/scene_node.h>
#include <base/texture.h>
#include <base/sampler.h>
#include <base/spectrum.h>
#include <base/interaction.h>

#include <utility>

namespace luisa::render {

using compute::BindlessArray;
using compute::Expr;
using compute::Float3;
using compute::Var;

class Shape;
class Sampler;
class Frame;
class Interaction;

class Surface : public SceneNode {

public:
    struct Evaluation {
        SampledSpectrum f;
        Float pdf;
        Float3 normal;
        Float2 roughness;
        SampledSpectrum eta;
        [[nodiscard]] static auto zero(size_t spec_dim) noexcept {
            return Evaluation{
                .f = SampledSpectrum{spec_dim},
                .pdf = 0.f,
                .normal = make_float3(0.f, 0.f, 1.f),
                .roughness = make_float2(),
                .eta = SampledSpectrum{spec_dim, 1.f}};
        }
    };

    struct Sample {
        Float3 wi;
        Evaluation eval;
        [[nodiscard]] static auto zero(size_t spec_dim) noexcept {
            return Sample{.wi = make_float3(0.f, 0.f, 1.f),
                          .eval = Evaluation::zero(spec_dim)};
        }
    };

    class Instance;

    class Closure {

    private:
        const Instance *_instance;

    protected:
        Interaction _it;
        const SampledWavelengths &_swl;
        Float _time;

    public:
        Closure(const Instance *instance, Interaction it,
                const SampledWavelengths &swl, Expr<float> time) noexcept;
        virtual ~Closure() noexcept = default;
        template<typename T = Instance>
            requires std::is_base_of_v<Instance, T>
        [[nodiscard]] auto instance() const noexcept { return static_cast<const T *>(_instance); }
        [[nodiscard]] virtual Evaluation evaluate(Expr<float3> wi) const noexcept = 0;
        [[nodiscard]] virtual Sample sample(Expr<float> u_lobe, Expr<float2> u) const noexcept = 0;
        [[nodiscard]] auto &interaction() const noexcept { return _it; }
        virtual void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept = 0;
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Surface *_surface;

    private:
        friend class Surface;
        const Texture::Instance *_alpha{nullptr};
        const Texture::Instance *_normal{nullptr};

    private:
        [[nodiscard]] virtual luisa::unique_ptr<Closure> _closure(
            const Interaction &it,
            const SampledWavelengths &swl,
            Expr<float> time) const noexcept = 0;

    public:
        Instance(const Pipeline &pipeline, const Surface *surface) noexcept
            : _pipeline{pipeline}, _surface{surface} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Surface>
            requires std::is_base_of_v<Surface, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_surface); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto alpha() const noexcept { return _alpha; }
        [[nodiscard]] auto normal() const noexcept { return _normal; }
        [[nodiscard]] luisa::unique_ptr<Closure> closure(
            Interaction it, const SampledWavelengths &swl, Expr<float> time) const noexcept;
    };

private:
    const Texture *_normal;
    const Texture *_alpha;

private:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

public:
    Surface(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto normal() const noexcept { return _normal; }
    [[nodiscard]] auto alpha() const noexcept { return _alpha; }
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

}// namespace luisa::render

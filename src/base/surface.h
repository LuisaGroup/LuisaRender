//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/spectrum.h>
#include <base/scene_node.h>
#include <base/texture.h>
#include <base/sampler.h>

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
        SampledWavelengths swl;
        Float4 f;
        Float pdf;
        Float2 alpha;
        Float4 eta;
    };

    struct Sample {
        Float3 wi;
        Evaluation eval;
    };

    struct Closure {
        virtual ~Closure() noexcept = default;
        [[nodiscard]] virtual Evaluation evaluate(Expr<float3> wi) const noexcept = 0;
        [[nodiscard]] virtual Sample sample(Sampler::Instance &sampler) const noexcept = 0;
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Surface *_surface;

    public:
        Instance(const Pipeline &pipeline, const Surface *surface) noexcept
            : _pipeline{pipeline}, _surface{surface} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Surface>
            requires std::is_base_of_v<Surface, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_surface); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual luisa::unique_ptr<Closure> closure(
            const Interaction &it,
            const SampledWavelengths &swl,
            Expr<float> time) const noexcept = 0;
    };

private:
    const Texture *_normal_map;

public:
    Surface(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto normal_map() const noexcept { return _normal_map; }
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] virtual uint /* material_id */ build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

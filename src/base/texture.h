//
// Created by Mike Smith on 2022/1/25.
//

#pragma once

#include <dsl/syntax.h>
#include <util/spec.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/scene_node.h>
#include <base/spectrum.h>

namespace luisa::render {

class Pipeline;
class Interaction;
class SampledWavelengths;

using compute::Buffer;
using compute::BufferView;
using compute::Float4;
using compute::Image;
using compute::PixelStorage;
using TextureSampler = compute::Sampler;

class Texture : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const Texture *_texture;

    protected:
        [[nodiscard]] Spectrum::Decode _evaluate_static_albedo_spectrum(
            const SampledWavelengths &swl, float4 v) const noexcept;
        [[nodiscard]] Spectrum::Decode _evaluate_static_unbounded_spectrum(
            const SampledWavelengths &swl, float4 v) const noexcept;
        [[nodiscard]] Spectrum::Decode _evaluate_static_illuminant_spectrum(
            const SampledWavelengths &swl, float4 v) const noexcept;

    public:
        Instance(const Pipeline &pipeline, const Texture *texture) noexcept
            : _pipeline{pipeline}, _texture{texture} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Texture>
            requires std::is_base_of_v<Texture, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_texture); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Float4 evaluate(
            const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum eval_grad(
            const Interaction &it, const SampledWavelengths &swl, Expr<float> time, Expr<float4> grad) const noexcept = 0;
        virtual void backward(
            const Interaction &it, const SampledWavelengths &swl, Expr<float> time, Expr<float4> grad) const noexcept = 0;
        [[nodiscard]] virtual Spectrum::Decode evaluate_albedo_spectrum(
            const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Spectrum::Decode evaluate_unbounded_spectrum(
            const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual Spectrum::Decode evaluate_illuminant_spectrum(
            const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] SampledSpectrum eval_grad_albedo_spectrum(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time, const SampledSpectrum &dSpec) const noexcept;
        void backward_albedo_spectrum(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time, const SampledSpectrum &dSpec) const noexcept;

        void backward_illuminant_spectrum(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time, const SampledSpectrum &dSpec) const noexcept;
        void backward_unbounded_spectrum(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time, const SampledSpectrum &dSpec) const noexcept;

    public:
        inline static const luisa::string non_differrentiable_identifier = "nodiff";
        [[nodiscard]] virtual luisa::string diff_param_identifier() const noexcept {
            return non_differrentiable_identifier;
        }
        [[nodiscard]] static auto diff_param_identifier(const Texture::Instance *t) noexcept {
            return t ? t->diff_param_identifier() : non_differrentiable_identifier;
        }
    };

private:
    float2 _range;
    bool _requires_grad;
    bool _render_grad_map;

public:
    Texture(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto range() const noexcept { return _range; }
    [[nodiscard]] virtual bool requires_gradients() const noexcept;
    [[nodiscard]] virtual bool render_grad_map() const noexcept;
    virtual void disable_gradients() noexcept;
    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual bool is_constant() const noexcept = 0;
    [[nodiscard]] virtual luisa::optional<float4> evaluate_static() const noexcept;
    [[nodiscard]] virtual uint channels() const noexcept { return 4u; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Texture::Instance)

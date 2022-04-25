//
// Created by Mike Smith on 2022/1/25.
//

#pragma once

#include <dsl/syntax.h>
#include <util/spec.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/scene_node.h>
#include <base/differentiation.h>

namespace luisa::render {

#define LUISA_RENDER_PARAM_CHANNEL_CHECK(class_name, name, channel_num) \
    [&] {                                                               \
        if ((_##name != nullptr) &&                                     \
            (_##name->channels() < channel_num##u)) [[unlikely]] {      \
            LUISA_ERROR_WITH_LOCATION(                                  \
                "Expected channels >= " #channel_num                    \
                " for " #class_name "::" #name ".");                    \
        }                                                               \
    }()

class Pipeline;
class Interaction;
class SampledWavelengths;

using compute::Buffer;
using compute::BufferView;
using compute::CommandBuffer;
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

    public:
        Instance(const Pipeline &pipeline, const Texture *texture) noexcept
            : _pipeline{pipeline}, _texture{texture} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Texture>
            requires std::is_base_of_v<Texture, T> [
                [nodiscard]] auto
            node() const noexcept { return static_cast<const T *>(_texture); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Float4 evaluate(const Interaction &it, Expr<float> time) const noexcept = 0;
        virtual void backward(const Interaction &it, Expr<float> time, Expr<float4> grad) const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum evaluate_albedo_spectrum(
            const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept;
        [[nodiscard]] virtual SampledSpectrum evaluate_illuminant_spectrum(
            const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept;
        void backward_albedo_spectrum(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time, const SampledSpectrum &dSpec) const noexcept;
        void backward_illuminant_spectrum(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time, const SampledSpectrum &dSpec) const noexcept;
    };

private:
    float2 _range;
    bool _requires_grad;

public:
    Texture(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto range() const noexcept { return _range; }
    [[nodiscard]] auto requires_gradients() const noexcept { return _requires_grad; }
    void disable_gradients() noexcept { _requires_grad = false; }
    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual bool is_constant() const noexcept = 0;
    [[nodiscard]] virtual uint channels() const noexcept { return 4u; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

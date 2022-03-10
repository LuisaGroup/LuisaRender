//
// Created by Mike Smith on 2022/1/25.
//

#pragma once

#include <dsl/syntax.h>
#include <util/spectrum.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/scene_node.h>
#include <base/differentiation.h>

namespace luisa::render {

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
    enum struct Category {
        COLOR,
        ILLUMINANT,
        GENERIC
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Texture *_texture;

    public:
        Instance(const Pipeline &pipeline, const Texture *texture) noexcept
            : _pipeline{pipeline}, _texture{texture} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Texture>
            requires std::is_base_of_v<Texture, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_texture); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual Float4 evaluate(
            const Interaction &it,
            const SampledWavelengths &swl,
            Expr<float> time) const noexcept = 0;
        virtual void backward(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time, Expr<float4> grad) const noexcept = 0;
    };

private:
    bool _requires_grad;

public:
    Texture(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto requires_gradients() const noexcept { return _requires_grad; }
    void disable_gradients() noexcept { _requires_grad = false; }
    [[nodiscard]] virtual bool is_black() const noexcept = 0;
    [[nodiscard]] virtual uint channels() const noexcept { return 4u; }
    [[nodiscard]] virtual Category category() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

class ImageTexture : public Texture {

public:
    class Instance : public Texture::Instance {

    private:
        luisa::optional<Differentiation::TexturedParameter> _diff_param;
        uint _texture_id;

    private:
        [[nodiscard]] Float2 _compute_uv(const Interaction &it) const noexcept;

    public:
        Instance(const Pipeline &pipeline, const Texture *texture, uint texture_id,
                 luisa::optional<Differentiation::TexturedParameter> param) noexcept
            : Texture::Instance{pipeline, texture},
              _texture_id{texture_id}, _diff_param{std::move(param)} {}
        [[nodiscard]] Float4 evaluate(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time) const noexcept override;
        void backward(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> time, Expr<float4> grad) const noexcept override;
    };

private:
    TextureSampler _sampler;
    float2 _uv_scale;
    float2 _uv_offset;

private:
    [[nodiscard]] virtual const LoadedImage &_image() const noexcept = 0;

public:
    ImageTexture(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto uv_scale() const noexcept { return _uv_scale; }
    [[nodiscard]] auto uv_offset() const noexcept { return _uv_offset; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] uint channels() const noexcept override { return _image().channels(); }
    [[nodiscard]] luisa::unique_ptr<Texture::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

}// namespace luisa::render

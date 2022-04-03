//
// Created by Mike Smith on 2022/3/9.
//

#pragma once

#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/shader.h>
#include <runtime/command_buffer.h>

namespace luisa::render {

using compute::Buffer;
using compute::BufferView;
using compute::CommandBuffer;
using compute::Expr;
using compute::Float4;
using compute::Image;
using compute::ImageView;
using compute::Shader1D;
using compute::Shader2D;
using TextureSampler = compute::Sampler;

class Pipeline;

class Differentiation {

public:
    static constexpr auto constant_parameter_buffer_capacity = 4096u;
    static constexpr auto gradiant_collision_avoidance_block_size = 256u;
    static constexpr auto constant_parameter_gradient_buffer_size =
        constant_parameter_buffer_capacity * 4u *
        gradiant_collision_avoidance_block_size;

    class ConstantParameter {

    private:
        uint _index;
        uint _channels;

    public:
        ConstantParameter(uint index, uint channels) noexcept
            : _index{index}, _channels{channels} {}
        [[nodiscard]] auto index() const noexcept { return _index; }
        [[nodiscard]] auto channels() const noexcept { return _channels; }
    };

    class TexturedParameter {

    private:
        const Image<float> &_image;
        TextureSampler _sampler;
        uint _grad_offset;

    public:
        TexturedParameter(const Image<float> &image, TextureSampler sampler, uint grad_offset) noexcept
            : _image{image}, _sampler{sampler}, _grad_offset{grad_offset} {}
        [[nodiscard]] auto &image() const noexcept { return _image; }
        [[nodiscard]] auto sampler() const noexcept { return _sampler; }
        [[nodiscard]] auto gradient_buffer_offset() const noexcept { return _grad_offset; }
    };

private:
    Pipeline &_pipeline;
    luisa::optional<BufferView<uint>> _grad_buffer;
    BufferView<float4> _const_param_buffer;
    uint _gradient_buffer_size;
    luisa::vector<float4> _constant_params;
    luisa::vector<TexturedParameter> _textured_params;
    Shader1D<Buffer<uint>> _clear_grad;
    Shader1D<Buffer<uint>, Buffer<float4>, float> _apply_grad_const;
    Shader2D<Buffer<uint>, uint, Image<float>, uint, float> _apply_grad_tex;

public:
    explicit Differentiation(Pipeline &pipeline) noexcept;
    [[nodiscard]] ConstantParameter parameter(float x) noexcept;
    [[nodiscard]] ConstantParameter parameter(float2 x) noexcept;
    [[nodiscard]] ConstantParameter parameter(float3 x) noexcept;
    [[nodiscard]] ConstantParameter parameter(float4 x) noexcept;
    [[nodiscard]] ConstantParameter parameter(float4 x, uint channels) noexcept;
    [[nodiscard]] TexturedParameter parameter(const Image<float> &image, TextureSampler s) noexcept;
    void materialize(CommandBuffer &command_buffer) noexcept;
    void clear_gradients(CommandBuffer &command_buffer) noexcept;
    void apply_gradients(CommandBuffer &command_buffer, float alpha) noexcept;
    /// Apply then clear the gradients
    void step(CommandBuffer &command_buffer, float alpha) noexcept;

public:
    [[nodiscard]] Float4 decode(const ConstantParameter &param) const noexcept;
    void accumulate(const ConstantParameter &param, Expr<float4> grad) const noexcept;
    void accumulate(const TexturedParameter &param, Expr<float2> p, Expr<float4> grad) const noexcept;
};

}// namespace luisa::render

//
// Created by Mike Smith on 2022/3/9.
//

#pragma once

#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/shader.h>
#include <runtime/command_buffer.h>

#include <util/optimizer.h>

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
    static constexpr auto gradiant_collision_avoidance_block_size = 512u;
    static constexpr auto constant_parameter_counter_size =
        constant_parameter_buffer_capacity *
        gradiant_collision_avoidance_block_size;
    static constexpr auto constant_parameter_gradient_buffer_size =
        constant_parameter_buffer_capacity * 4u *
        gradiant_collision_avoidance_block_size;

    static constexpr auto constant_min_count = 1.f;

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
        uint _counter_offset;
        float2 _range;

    public:
        TexturedParameter(const Image<float> &image, TextureSampler sampler, uint grad_offset,
                          uint counter_offset, float2 range) noexcept
            : _image{image}, _sampler{sampler}, _grad_offset{grad_offset}, _counter_offset{counter_offset},
              _range{range} {}
        [[nodiscard]] auto &image() const noexcept { return _image; }
        [[nodiscard]] auto sampler() const noexcept { return _sampler; }
        [[nodiscard]] auto range() const noexcept { return _range; }
        [[nodiscard]] auto gradient_buffer_offset() const noexcept { return _grad_offset; }
        [[nodiscard]] auto counter_offset() const noexcept { return _counter_offset; }
    };

private:
    Pipeline &_pipeline;

    Optimizer _optimizer;

    luisa::vector<float4> _constant_params;
    luisa::vector<float2> _constant_ranges;
    luisa::vector<TexturedParameter> _textured_params;

    BufferView<float4> _const_param_buffer;
    BufferView<float2> _const_param_range_buffer;

    uint _gradient_buffer_size;
    luisa::optional<BufferView<uint>> _grad_buffer;

    uint _counter_size;
    luisa::optional<BufferView<uint>> _counter;

    Shader1D<Buffer<uint>> _clear_buffer;
    Shader1D<Buffer<uint>, Buffer<float4>, Buffer<float2>, float, Buffer<uint>> _apply_grad_const;
    Shader2D<Buffer<uint>, uint, Image<float>, uint, float, float2, Buffer<uint>, uint> _apply_grad_tex;

public:
    explicit Differentiation(Pipeline &pipeline, const Optimizer &optimizer) noexcept;
    [[nodiscard]] ConstantParameter parameter(float x, float2 range) noexcept;
    [[nodiscard]] ConstantParameter parameter(float2 x, float2 range) noexcept;
    [[nodiscard]] ConstantParameter parameter(float3 x, float2 range) noexcept;
    [[nodiscard]] ConstantParameter parameter(float4 x, float2 range) noexcept;
    [[nodiscard]] ConstantParameter parameter(float4 x, uint channels, float2 range) noexcept;
    [[nodiscard]] TexturedParameter parameter(const Image<float> &image, TextureSampler s, float2 range) noexcept;
    void materialize(CommandBuffer &command_buffer) noexcept;
    void clear_gradients(CommandBuffer &command_buffer) noexcept;
    void apply_gradients(CommandBuffer &command_buffer, float alpha) noexcept;
    /// Apply then clear the gradients
    void step(CommandBuffer &command_buffer, float alpha) noexcept;
    void dump(CommandBuffer &command_buffer, const std::filesystem::path &folder) const noexcept;

public:
    [[nodiscard]] Float4 decode(const ConstantParameter &param) const noexcept;
    void accumulate(const ConstantParameter &param, Expr<float4> grad) const noexcept;
    void accumulate(const TexturedParameter &param, Expr<float2> p, Expr<float4> grad) const noexcept;
};

}// namespace luisa::render

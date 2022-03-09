//
// Created by Mike Smith on 2022/3/9.
//

#include <dsl/syntax.h>
#include <base/pipeline.h>
#include <base/differentiation.h>
#include <util/atomic.h>

namespace luisa::render {

Differentiation::Differentiation(Pipeline &pipeline) noexcept
    : _pipeline{pipeline},
      _const_param_buffer{*pipeline.create<Buffer<float4>>(constant_parameter_buffer_capacity)},
      _gradient_buffer_size{constant_parameter_buffer_capacity * 4u} {

    using namespace compute;
    Kernel1D clear_grad = [](BufferUInt gradients) noexcept {
        gradients.write(dispatch_x(), 0u);
    };
    Kernel1D apply_grad_const = [](BufferUInt gradients, BufferFloat4 params, Float alpha) noexcept {
        auto i = dispatch_x();
        auto x = as<float>(gradients.read(i * 4u + 0u));
        auto y = as<float>(gradients.read(i * 4u + 1u));
        auto z = as<float>(gradients.read(i * 4u + 2u));
        auto w = as<float>(gradients.read(i * 4u + 3u));
        auto grad = make_float4(x, y, z, w);
        auto old = params.read(i);
        params.write(i, fma(-alpha, grad, old));
    };
    Kernel2D apply_grad_tex = [](BufferUInt gradients, UInt offset, ImageFloat image, UInt channels, Float alpha) noexcept {
        auto coord = dispatch_id().xy();
        auto i = coord.y * dispatch_size_x() + coord.x;
        auto x = as<float>(gradients.read(offset + i * channels + 0u));
        auto y = as<float>(gradients.read(offset + i * channels + 1u));
        auto z = as<float>(gradients.read(offset + i * channels + 2u));
        auto w = as<float>(gradients.read(offset + i * channels + 3u));
        auto grad = make_float4(x, y, z, w);
        auto old = image.read(coord);
        image.write(coord, fma(-alpha, grad, old));
    };
    _clear_grad = _pipeline.device().compile(clear_grad);
    _apply_grad_const = _pipeline.device().compile(apply_grad_const);
    _apply_grad_tex = _pipeline.device().compile(apply_grad_tex);
}

Differentiation::ConstantParameter Differentiation::parameter(float4 x, uint channels) noexcept {
    LUISA_ASSERT(
        _constant_params.size() < constant_parameter_buffer_capacity,
        "Too many parameters in differentiation.");
    auto index = static_cast<uint>(_constant_params.size());
    _constant_params.emplace_back(x);
    return {index, channels};
}

Differentiation::ConstantParameter Differentiation::parameter(float x) noexcept {
    return parameter(make_float4(x, 0.f, 0.f, 0.f), 1u);
}

Differentiation::ConstantParameter Differentiation::parameter(float2 x) noexcept {
    return parameter(make_float4(x, 0.f, 0.f), 2u);
}

Differentiation::ConstantParameter Differentiation::parameter(float3 x) noexcept {
    return parameter(make_float4(x, 0.f), 3u);
}
Differentiation::ConstantParameter Differentiation::parameter(float4 x) noexcept {
    return parameter(x, 4u);
}

Differentiation::TexturedParameter Differentiation::parameter(const Image<float> &image, TextureSampler s) noexcept {
    auto nc = compute::pixel_format_channel_count(image.format());
    auto param_count = image.size().x * image.size().y * nc;
    auto grad_offset = _gradient_buffer_size;
    _gradient_buffer_size = (_gradient_buffer_size + param_count - 3u) / 4u * 4u;
    TexturedParameter param{image, s, grad_offset};
    return _textured_params.emplace_back(param);
}

void Differentiation::materialize(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(!_grad_buffer, "Differentiation already materialized.");
    _grad_buffer.emplace(*_pipeline.create<Buffer<uint>>(_gradient_buffer_size));
    auto n = _constant_params.size();
    command_buffer << _const_param_buffer.subview(0u, n).copy_from(_constant_params.data());
    clear_gradients(command_buffer);
}

void Differentiation::clear_gradients(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    command_buffer << _clear_grad(*_grad_buffer).dispatch(_gradient_buffer_size);
}

void Differentiation::apply_gradients(CommandBuffer &command_buffer, float alpha) noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    command_buffer << _apply_grad_const(*_grad_buffer, _const_param_buffer, alpha)
                          .dispatch(_constant_params.size());
    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto offset = p.gradient_buffer_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        command_buffer << _apply_grad_tex(*_grad_buffer, offset, image, channels, alpha)
                              .dispatch(image.size())
                       << compute::commit();
    }
}

Float4 Differentiation::decode(const Differentiation::ConstantParameter &param) const noexcept {
    return _const_param_buffer.read(param.index());
}

void Differentiation::accumulate(const Differentiation::ConstantParameter &param, Expr<float4> grad) const noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    for (auto i = 0u; i < param.channels(); i++) {
        atomic_float_add(*_grad_buffer, param.index() * 4u + i, grad[i]);
    }
}

void Differentiation::accumulate(const Differentiation::TexturedParameter &param, Expr<float2> p, Expr<float4> grad) const noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    // TODO
}

}// namespace luisa::render

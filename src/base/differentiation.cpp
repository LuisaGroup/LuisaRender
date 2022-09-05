//
// Created by Mike Smith on 2022/3/9.
//

#include <tinyexr.h>

#include <dsl/syntax.h>
#include <base/pipeline.h>
#include <base/differentiation.h>
#include <util/atomic.h>
#include <util/rng.h>

namespace luisa::render {

Differentiation::Differentiation(Pipeline &pipeline) noexcept
    : _pipeline{pipeline},
      //      _const_param_buffer{*pipeline.create<Buffer<float4>>(constant_parameter_buffer_capacity)},
      //      _const_param_range_buffer{*pipeline.create<Buffer<float2>>(constant_parameter_buffer_capacity)},
      _gradient_buffer_size{constant_parameter_gradient_buffer_size},
      _param_buffer_size{constant_parameter_buffer_capacity * 4u},
      _counter_size{constant_parameter_counter_size} {

    _constant_params.reserve(constant_parameter_buffer_capacity);
    _constant_ranges.reserve(constant_parameter_buffer_capacity);

    using namespace compute;
    Kernel1D clear_buffer = [](BufferUInt gradients) noexcept {
        gradients.write(dispatch_x(), 0u);
    };
    _clear_buffer = _pipeline.device().compile(clear_buffer);

    Kernel1D accumulate_grad_const_kernel = [](BufferUInt gradients, BufferFloat param_gradients, BufferUInt counter) noexcept {
        auto thread = dispatch_x();
        auto counter_offset = thread * gradiant_collision_avoidance_block_size;
        auto grad_offset = 4u * counter_offset;
        auto grad = def(make_float4());
        auto count = def(0u);
        for (auto i = 0u; i < gradiant_collision_avoidance_block_size; i++) {
            auto x = as<float>(gradients.read(grad_offset + i * 4u + 0u));
            auto y = as<float>(gradients.read(grad_offset + i * 4u + 1u));
            auto z = as<float>(gradients.read(grad_offset + i * 4u + 2u));
            auto w = as<float>(gradients.read(grad_offset + i * 4u + 3u));
            grad += make_float4(x, y, z, w);
            count += counter.read(counter_offset + i);
        }
        grad /= Float(max(count, 1u));
        auto param_offset = thread * 4u;
        param_gradients.write(param_offset + 0u, grad.x);
        param_gradients.write(param_offset + 1u, grad.y);
        param_gradients.write(param_offset + 2u, grad.z);
        param_gradients.write(param_offset + 3u, grad.w);
    };
    _accumulate_grad_const = _pipeline.device().compile(accumulate_grad_const_kernel);

    Kernel1D accumulate_grad_tex_kernel = [](BufferUInt gradients, UInt grad_offset,
                                             BufferUInt counter, UInt counter_offset,
                                             BufferFloat param_gradients, UInt param_offset) noexcept {
        auto index = dispatch_x();
        auto grad = as<float>(gradients.read(grad_offset + index));
        auto count = counter.read(counter_offset + index);
        grad /= Float(max(count, 1u));
        param_gradients.write(param_offset + index, grad);
    };
    _accumulate_grad_tex = _pipeline.device().compile(accumulate_grad_tex_kernel);
}

Differentiation::ConstantParameter Differentiation::parameter(float4 x, uint channels, float2 range) noexcept {
    auto index = static_cast<uint>(_constant_params.size());
    _constant_params.emplace_back(x);
    _constant_ranges.emplace_back(range);
    return {index, channels};
}

Differentiation::ConstantParameter Differentiation::parameter(float x, float2 range) noexcept {
    return parameter(make_float4(x, 0.f, 0.f, 0.f), 1u, range);
}

Differentiation::ConstantParameter Differentiation::parameter(float2 x, float2 range) noexcept {
    return parameter(make_float4(x, 0.f, 0.f), 2u, range);
}

Differentiation::ConstantParameter Differentiation::parameter(float3 x, float2 range) noexcept {
    return parameter(make_float4(x, 0.f), 3u, range);
}
Differentiation::ConstantParameter Differentiation::parameter(float4 x, float2 range) noexcept {
    return parameter(x, 4u, range);
}

Differentiation::TexturedParameter Differentiation::parameter(const Image<float> &image, TextureSampler s, float2 range) noexcept {
    auto nc = compute::pixel_format_channel_count(image.format());
    auto pixel_count = image.size().x * image.size().y;
    auto param_count = pixel_count * nc;
    auto grad_offset = _gradient_buffer_size;
    auto param_offset = _param_buffer_size;
    auto counter_offset = _counter_size;
    _counter_size = (_counter_size + pixel_count + 3u) & ~0b11u;
    _param_buffer_size = (_param_buffer_size + param_count + 3u) & ~0b11u;
    _gradient_buffer_size = (_gradient_buffer_size + param_count + 3u) & ~0b11u;
    return _textured_params.emplace_back(TexturedParameter{image, s, grad_offset, param_offset, counter_offset, range});
}

void Differentiation::materialize(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(!_grad_buffer, "Differentiation already materialized.");
    _param_buffer_size = _gradient_buffer_size - constant_parameter_gradient_buffer_size + constant_parameter_buffer_capacity * 4u;
    _param_buffer.emplace(*_pipeline.create<Buffer<float>>(std::max(_param_buffer_size, 1u)));
    _param_range_buffer.emplace(*_pipeline.create<Buffer<float2>>(std::max(_param_buffer_size, 1u)));
    _param_grad_buffer.emplace(*_pipeline.create<Buffer<float>>(std::max(_param_buffer_size, 1u)));
    _grad_buffer.emplace(*_pipeline.create<Buffer<uint>>(std::max(_gradient_buffer_size, 1u)));
    _counter.emplace(*_pipeline.create<Buffer<uint>>(std::max(_counter_size, 1u)));
    clear_gradients(command_buffer);

    if (auto n = _constant_params.size()) {
        //        command_buffer << _const_param_buffer.subview(0u, n)
        //                              .copy_from(_constant_params.data())
        //                       << _const_param_range_buffer.subview(0u, n)
        //                              .copy_from(_constant_ranges.data());
        command_buffer << _param_buffer.value().subview(0u, 4u * n).copy_from(_constant_params.data())
                       << _param_range_buffer.value().subview(0u, n).copy_from(_constant_ranges.data());
    }

    Kernel1D textured_params_range_kernel = [](BufferFloat2 params_range_buffer, Float2 range, UInt start) noexcept {
        params_range_buffer.write(start + dispatch_x(), range);
    };
    auto textured_params_range_shader = _pipeline.device().compile(textured_params_range_kernel);

    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto param_offset = p.param_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        auto length = image.size().x * image.size().y * channels;
        command_buffer << image.copy_to(_param_buffer.value().subview(param_offset, length))
                       << textured_params_range_shader(*_param_range_buffer, p.range(), param_offset).dispatch(length);
    }
    command_buffer << synchronize();

    _optimizer->initialize(command_buffer, _param_buffer_size, *_param_buffer, *_param_grad_buffer, *_param_range_buffer);
}

void Differentiation::clear_gradients(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    if (auto n = _gradient_buffer_size) {
        command_buffer << _clear_buffer(*_grad_buffer).dispatch(n);
    }
    if (auto n = _counter_size) {
        command_buffer << _clear_buffer(*_counter).dispatch(n);
    }
}

void Differentiation::apply_gradients(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");

    // accumulate constant parameters
    if (auto n = _constant_params.size()) {
        command_buffer << _accumulate_grad_const(*_grad_buffer, *_param_grad_buffer, *_counter).dispatch(n);

        //        luisa::vector<float4> params_before(n);
        //        luisa::vector<uint> counter(n * gradiant_collision_avoidance_block_size);
        //        luisa::vector<float4> collision_avoiding_gradients(n * gradiant_collision_avoidance_block_size);
        //        luisa::vector<float4> params_after(n);
        //        command_buffer << _const_param_buffer.subview(0u, n).copy_to(params_before.data())
        //                       << _counter->subview(0u, n * gradiant_collision_avoidance_block_size).copy_to(counter.data())
        //                       << _grad_buffer->subview(0u, n * 4u * gradiant_collision_avoidance_block_size)
        //                              .copy_to(collision_avoiding_gradients.data())
        //                       << _accumulate_grad_const(*_grad_buffer, *_counter).dispatch(n);
    }

    // accumulate textured parameters
    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto param_offset = p.param_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        auto length = image.size().x * image.size().y * channels;
        command_buffer << image.copy_from(_param_buffer.value().subview(param_offset, length));
    }

    _optimizer->step(command_buffer);

    //    if (auto n = _constant_params.size()) {
    //        command_buffer << _apply_grad_const(*_grad_buffer, _const_param_buffer, _const_param_range_buffer, alpha, *_counter)
    //                              .dispatch(n)
    //                       << _const_param_buffer.subview(0u, n).copy_to(params_after.data())
    //                       << compute::synchronize();
    //
    //        // DEBUG: print constant parameters
    //        for (auto i = 0u; i < n; i++) {
    //            auto p0 = params_before[i];
    //            auto grad = make_float4();
    //            auto count_uint = 0u;
    //            for (auto g = 0u; g < gradiant_collision_avoidance_block_size; g++) {
    //                auto index = i * gradiant_collision_avoidance_block_size + g;
    //                grad += collision_avoiding_gradients[index];
    //                count_uint += counter[index];
    //            }
    //            auto count = float(std::max(count_uint, 1u));
    //            grad /= count;
    //            auto p1 = params_after[i];
    //            LUISA_INFO(
    //                "Param #{}: ({}, {}, {}, {}) - "
    //                "{} * ({}, {}, {}, {}) -> "
    //                "({}, {}, {}, {})"
    //                ", count = {}",
    //                i, p0.x, p0.y, p0.z, p0.w,
    //                alpha, grad.x, grad.y, grad.z, grad.w,
    //                p1.x, p1.y, p1.z, p1.w,
    //                count);
    //        }
    //    }

    // apply textured parameters
    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto param_offset = p.param_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        auto length = image.size().x * image.size().y * channels;
        command_buffer << image.copy_from(_param_buffer.value().subview(param_offset, length));
    }
}

Float4 Differentiation::decode(const Differentiation::ConstantParameter &param) const noexcept {
    //    return _const_param_buffer.read(param.index());
    auto x = _param_buffer.value().read(param.index() * 4u + 0u);
    auto y = _param_buffer.value().read(param.index() * 4u + 1u);
    auto z = _param_buffer.value().read(param.index() * 4u + 2u);
    auto w = _param_buffer.value().read(param.index() * 4u + 3u);
    return make_float4(x, y, z, w);
}

void Differentiation::accumulate(const Differentiation::ConstantParameter &param, Expr<float4> grad,
                                 Expr<uint> slot_seed) const noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    auto slots = (slot_seed ^ pcg4d(as<uint4>(grad))) & gradiant_collision_avoidance_bit_and;
    for (auto i = 0u; i < param.channels(); i++) {
        auto grad_offset = (param.index() * gradiant_collision_avoidance_block_size + slots[i]) * 4u + i;
        atomic_float_add(*_grad_buffer, grad_offset, grad[i]);
    }
    auto counter_offset = param.index() * gradiant_collision_avoidance_block_size + slots[0];
    _counter->atomic(counter_offset).fetch_add(1u);
}

void Differentiation::accumulate(const Differentiation::TexturedParameter &param, Expr<float2> p,
                                 Expr<float4> grad) const noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    using namespace compute;
    auto map_uv = [s = param.sampler()](Expr<float2> uv) noexcept {
        switch (s.address()) {
            case TextureSampler::Address::EDGE:
                return clamp(uv, 0.f, one_minus_epsilon);
            case TextureSampler::Address::REPEAT:
                return fract(uv);
            case TextureSampler::Address::MIRROR: {
                auto t = floor(uv);
                auto frac = uv - t;
                return ite(make_int2(t) % 2 == 0, frac, 1.f - frac);
            }
            case TextureSampler::Address::ZERO:
                return def(uv);
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture address mode.");
    };
    auto write_grad = [&param, this](Expr<float2> uv, Expr<float4> grad) noexcept {
        $if(all(uv >= 0.f && uv < 1.f)) {
            auto size = param.image().size();
            auto coord = clamp(make_uint2(uv * make_float2(size)), 0u, size - 1u);
            auto pixel_id = coord.y * size.x + coord.x;
            auto nc = pixel_format_channel_count(param.image().format());
            auto grad_offset = param.gradient_buffer_offset() + pixel_id * nc;
            auto counter_offset = param.counter_offset() + pixel_id;
            for (auto i = 0u; i < nc; i++) {
                atomic_float_add(*_grad_buffer, grad_offset + i, grad[i]);
            }
            _counter->atomic(counter_offset).fetch_add(1u);
        };
    };
    write_grad(map_uv(p), grad);
}

void Differentiation::step(CommandBuffer &command_buffer) noexcept {
    apply_gradients(command_buffer);
    clear_gradients(command_buffer);
}

void Differentiation::dump(CommandBuffer &command_buffer, const std::filesystem::path &folder) const noexcept {
    // FIXME : several channels will be 0 when grads explode
    for (auto i = 0u; i < _textured_params.size(); i++) {
        auto param = _textured_params[i];
        auto image = param.image().view();
        auto size = image.size();
        auto channels = compute::pixel_storage_channel_count(image.storage());
        luisa::vector<float> pixels(size.x * size.y * channels);
        command_buffer << image.copy_to(pixels.data()) << compute::synchronize();
        auto file_name = folder / luisa::format("dump-{:05}.exr", i);
        SaveEXR(pixels.data(), static_cast<int>(size.x), static_cast<int>(size.y),
                static_cast<int>(channels), false, file_name.string().c_str(), nullptr);
    }
}

void Differentiation::register_optimizer(Optimizer::Instance *optimizer) noexcept {
    _optimizer = optimizer;
}

}// namespace luisa::render

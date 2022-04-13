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

Differentiation::Differentiation(Pipeline &pipeline, const Optimizer &optimizer) noexcept
    : _pipeline{pipeline},
      _optimizer{optimizer},
      _const_param_buffer{*pipeline.create<Buffer<float4>>(constant_parameter_buffer_capacity)},
      _const_param_range_buffer{*pipeline.create<Buffer<float2>>(constant_parameter_buffer_capacity)},
      _gradient_buffer_size{constant_parameter_gradient_buffer_size},
      _counter_size{constant_parameter_counter_size} {
    _constant_params.reserve(constant_parameter_buffer_capacity);
    _constant_ranges.reserve(constant_parameter_buffer_capacity);

    using namespace compute;
    Kernel1D clear_buffer = [](BufferUInt gradients) noexcept {
        gradients.write(dispatch_x(), 0u);
    };
    Kernel1D apply_grad_const = [](BufferUInt gradients, BufferFloat4 params, BufferFloat2 ranges,
                                   Float alpha, BufferUInt counter) noexcept {
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
        grad /= max(Float(count), constant_min_count);
        auto old = params.read(thread);
        auto range = ranges.read(thread);
        auto next = fma(-alpha, grad, old);
        params.write(thread, clamp(next, range.x, range.y));
    };
    Kernel2D apply_grad_tex = [](BufferUInt gradients, UInt grad_offset, ImageFloat image,
                                 UInt channels, Float alpha, Float2 range, BufferUInt counter, UInt counter_offset) noexcept {
        auto coord = dispatch_id().xy();
        auto i = coord.y * dispatch_size_x() + coord.x;
        auto x = as<float>(gradients.read(grad_offset + i * channels + 0u));
        auto y = as<float>(gradients.read(grad_offset + i * channels + 1u));
        auto z = as<float>(gradients.read(grad_offset + i * channels + 2u));
        auto w = as<float>(gradients.read(grad_offset + i * channels + 3u));
        auto grad = make_float4(x, y, z, w);
        grad /= max(Float(counter.read(counter_offset + i)), constant_min_count);
        auto old = image.read(coord);
        auto next = fma(-alpha, grad, old);
        image.write(coord, clamp(next, range.x, range.y));
    };
    _clear_buffer = _pipeline.device().compile(clear_buffer);
    _apply_grad_const = _pipeline.device().compile(apply_grad_const);
    _apply_grad_tex = _pipeline.device().compile(apply_grad_tex);
}

Differentiation::ConstantParameter Differentiation::parameter(float4 x, uint channels, float2 range) noexcept {
    LUISA_ASSERT(
        _constant_params.size() < constant_parameter_buffer_capacity,
        "Too many parameters in differentiation.");
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
    auto counter_offset = _counter_size;
    _gradient_buffer_size = (_gradient_buffer_size + param_count + 3u) & ~0b11u;
    _counter_size = (_counter_size + pixel_count + 3u) & ~0b11u;
    TexturedParameter param{image, s, grad_offset, counter_offset, range};
    return _textured_params.emplace_back(param);
}

void Differentiation::materialize(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(!_grad_buffer, "Differentiation already materialized.");
    _grad_buffer.emplace(*_pipeline.create<Buffer<uint>>(std::max(_gradient_buffer_size, 1u)));
    _counter.emplace(*_pipeline.create<Buffer<uint>>(std::max(_counter_size, 1u)));
    if (auto n = _constant_params.size()) {
        command_buffer << _const_param_buffer.subview(0u, n)
                              .copy_from(_constant_params.data())
                       << _const_param_range_buffer.subview(0u, n)
                              .copy_from(_constant_ranges.data());
        clear_gradients(command_buffer);
    }
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

void Differentiation::apply_gradients(CommandBuffer &command_buffer, float alpha) noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    if (auto n = _constant_params.size()) {
        luisa::vector<float4> params_before(n);
        luisa::vector<uint> counter(n * gradiant_collision_avoidance_block_size);
        luisa::vector<float4> collision_avoiding_gradients(n * gradiant_collision_avoidance_block_size);
        luisa::vector<float4> params_after(n);
        command_buffer << _const_param_buffer.subview(0u, n).copy_to(params_before.data())
                       << _counter->subview(0u, n * gradiant_collision_avoidance_block_size).copy_to(counter.data())
                       << _grad_buffer->subview(0u, n * 4u * gradiant_collision_avoidance_block_size)
                              .copy_to(collision_avoiding_gradients.data())
                       << _apply_grad_const(*_grad_buffer, _const_param_buffer, _const_param_range_buffer, alpha, *_counter)
                              .dispatch(n)
                       << _const_param_buffer.subview(0u, n).copy_to(params_after.data())
                       << compute::synchronize();
        for (auto i = 0u; i < n; i++) {
            auto p0 = params_before[i];
            auto grad = make_float4();
            auto count_uint = 0u;
            for (auto g = 0u; g < gradiant_collision_avoidance_block_size; g++) {
                auto index = i * gradiant_collision_avoidance_block_size + g;
                grad += collision_avoiding_gradients[index];
                count_uint += counter[index];
            }
            auto count = std::max(float(count_uint), constant_min_count);
            grad /= count;
            auto p1 = params_after[i];
            LUISA_INFO(
                "Param #{}: ({}, {}, {}, {}) - "
                "{} * ({}, {}, {}, {}) -> "
                "({}, {}, {}, {})"
                ", count = {}",
                i, p0.x, p0.y, p0.z, p0.w,
                alpha, grad.x, grad.y, grad.z, grad.w,
                p1.x, p1.y, p1.z, p1.w,
                count);
        }
    }
    for (auto &&p : _textured_params) {
        auto image = p.image().view();
        auto grad_offset = p.gradient_buffer_offset();
        auto counter_offset = p.counter_offset();
        auto channels = compute::pixel_format_channel_count(image.format());
        command_buffer << _apply_grad_tex(*_grad_buffer, grad_offset, image, channels, alpha,
                                          p.range(), *_counter, counter_offset)
                              .dispatch(image.size());
    }
}

Float4 Differentiation::decode(const Differentiation::ConstantParameter &param) const noexcept {
    return _const_param_buffer.read(param.index());
}

void Differentiation::accumulate(const Differentiation::ConstantParameter &param, Expr<float4> grad) const noexcept {
    LUISA_ASSERT(_grad_buffer, "Gradient buffer is not materialized.");
    auto bs = gradiant_collision_avoidance_block_size;
    auto slots = pcg4d(as<uint4>(grad)) % bs;
    for (auto i = 0u; i < param.channels(); i++) {
        auto grad_offset = (param.index() * bs + slots[i]) * 4u + i;
        atomic_float_add(*_grad_buffer, grad_offset, grad[i]);
    }
    auto counter_offset = param.index() * bs + slots[0];
    _counter->atomic(counter_offset).fetch_add(1u);
}

void Differentiation::accumulate(const Differentiation::TexturedParameter &param, Expr<float2> p, Expr<float4> grad) const noexcept {
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
            auto st = clamp(make_uint2(uv * make_float2(size)), 0u, size - 1u);
            auto pixel_id = st.y * size.x + st.x;
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

void Differentiation::step(CommandBuffer &command_buffer, float alpha) noexcept {
    apply_gradients(command_buffer, alpha);
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

}// namespace luisa::render

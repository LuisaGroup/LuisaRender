//
// Created by ChenXin on 2022/5/5.
//

#include <base/optimizer.h>
#include <base/pipeline.h>

namespace luisa::render {

Optimizer::Optimizer(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::OPTIMIZER},
      _learning_rate{std::max(desc->property_float_or_default("learning_rate", 0.1f), 0.f)} {}

Optimizer::Instance::Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Optimizer *optimizer) noexcept
    : _pipeline{pipeline}, _optimizer{optimizer} {

    Kernel1D clear_uint_kernel = [](BufferUInt buffer) {
        buffer.write(dispatch_x(), 0u);
    };
    _clear_uint_buffer = _pipeline.device().compile(clear_uint_kernel);

    Kernel1D clear_float_kernel = [](BufferFloat buffer) {
        buffer.write(dispatch_x(), 0.f);
    };
    _clear_float_buffer = _pipeline.device().compile(clear_float_kernel);

    Kernel1D clamp_range_kernel = [](BufferFloat gradients, BufferFloat params, BufferFloat2 ranges, Float alpha) noexcept {
        auto offset = dispatch_x();
        auto grad = gradients.read(offset);
        auto range = ranges.read(offset);
        auto old = params.read(offset);
        auto max_step_length = 0.1f * (range.y - range.x);
        grad = clamp(alpha * grad, -max_step_length, max_step_length);
        auto next = clamp(old - grad, range.x, range.y);
        params.write(offset, next);
    };
    _clamp_range = _pipeline.device().compile(clamp_range_kernel);
}

void Optimizer::Instance::initialize(CommandBuffer &command_buffer, uint length, BufferView<float> xi,
                                     BufferView<float> gradients, BufferView<float2> ranges) noexcept {
    _length = length;

    _ranges.reset();
    _xi.reset();
    _gradients.reset();

    _ranges.emplace(ranges);
    _xi.emplace(xi);
    _gradients.emplace(gradients);
}

void Optimizer::Instance::clamp_range(CommandBuffer &command_buffer) noexcept {
    auto learning_rate = node<Optimizer>()->learning_rate();
    command_buffer << _clamp_range(*_gradients, *_xi, *_ranges, learning_rate).dispatch(_length);
}

}// namespace luisa::render
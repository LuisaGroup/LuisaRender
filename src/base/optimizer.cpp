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
}

void Optimizer::Instance::initialize(CommandBuffer &command_buffer, uint length, BufferView<float> x0) noexcept {
    _length = length;
}

}// namespace luisa::render
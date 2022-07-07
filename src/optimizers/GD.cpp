//
// Created by Kasumi on 2022/7/6.
//

#include <base/optimizer.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class GD final : public Optimizer {

public:
    GD(Scene *scene, const SceneNodeDesc *desc)
    noexcept : Optimizer{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Optimizer::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class GDInstance final : public Optimizer::Instance {

private:
    Shader1D<Buffer<float>, Buffer<uint>, float> _update_params;

public:
    GDInstance(Pipeline &pipeline, CommandBuffer &command_buffer, const GD *optimizer) noexcept
        : Optimizer::Instance{pipeline, command_buffer, optimizer} {

        Kernel1D update_params_kernel = [](BufferFloat params, BufferUInt gradients, Float alpha) {
            auto index = dispatch_x();
            auto x = params.read(index);
            auto grad = as<float>(gradients.read(index));
            params.write(index, x - alpha * grad);
        };
        _update_params = pipeline.device().compile(update_params_kernel);
    }

public:
    void initialize(CommandBuffer &command_buffer, uint length, BufferView<float> x0) noexcept override;
    void step(CommandBuffer &command_buffer, BufferView<float> xi, BufferView<uint> gradients) noexcept override;
};

void GDInstance::initialize(CommandBuffer &command_buffer, uint length, BufferView<float> x0) noexcept {
    Optimizer::Instance::initialize(command_buffer, length, x0);
}

void GDInstance::step(CommandBuffer &command_buffer, BufferView<float> xi, BufferView<uint> gradients) noexcept {
    LUISA_ASSERT(_length != -1u, "Optimizer is not initialized.");
    command_buffer << _update_params(xi, gradients, node<GD>()->learning_rate()).dispatch(_length);
}

luisa::unique_ptr<Optimizer::Instance> GD::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<GDInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GD)
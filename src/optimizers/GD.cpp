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

public:
    GDInstance(Pipeline &pipeline, CommandBuffer &command_buffer, const GD *optimizer) noexcept
        : Optimizer::Instance{pipeline, command_buffer, optimizer} {}

public:
    void step(CommandBuffer &command_buffer) noexcept override;
};

void GDInstance::step(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(_length != -1u, "Optimizer is not initialized.");
    clamp_range(command_buffer);
}

luisa::unique_ptr<Optimizer::Instance> GD::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<GDInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GD)
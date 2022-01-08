//
// Created by Mike on 2022/1/7.
//

#include <scene/pipeline.h>
#include <scene/integrator.h>

namespace luisa::render {

class NormalVisualizer;

class NormalVisualizerInstance : public Integrator::Instance {

private:


public:
    explicit NormalVisualizerInstance(const NormalVisualizer *integrator) noexcept;

    void render(Stream &stream, Pipeline &pipeline) noexcept override {
//        for (auto i = 0; i < pipeline)
    }
};

class NormalVisualizer final : public Integrator {

public:
    NormalVisualizer(Scene *scene, const SceneNodeDesc *desc) noexcept : Integrator{scene, desc} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "normal"; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<NormalVisualizerInstance>(this);
    }
};

NormalVisualizerInstance::NormalVisualizerInstance(const NormalVisualizer *integrator) noexcept
    : Integrator::Instance{integrator} {}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalVisualizer)

//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <scene/light_sampler.h>

namespace luisa::render {

class UniformLightSampler final : public LightSampler {

public:
    unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    UniformLightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : LightSampler{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return "uniform"; }
};

unique_ptr<LightSampler::Instance> UniformLightSampler::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return nullptr;
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::UniformLightSampler)

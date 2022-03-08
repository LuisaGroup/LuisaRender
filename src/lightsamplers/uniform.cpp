//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <base/light_sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class UniformLightSampler final : public LightSampler {

public:
    luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    UniformLightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : LightSampler{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

class UniformLightSamplerInstance final : public LightSampler::Instance {

private:
    uint _light_buffer_id{};

public:
    UniformLightSamplerInstance(const LightSampler *sampler, Pipeline &pipeline, CommandBuffer &command_buffer) noexcept
        : LightSampler::Instance{pipeline, sampler} {
        auto [view, buffer_id] = pipeline.arena_buffer<Light::Handle>(pipeline.lights().size());
        _light_buffer_id = buffer_id;
        command_buffer << view.copy_from(pipeline.instanced_lights().data())
                       << compute::commit();
    }
    void update(CommandBuffer &, float) noexcept override {}
    [[nodiscard]] Float pmf(const Interaction &it) const noexcept {
        return static_cast<float>(1.0 / static_cast<double>(pipeline().lights().size()));
    }
    [[nodiscard]] Float pmf(const Interaction &it, const SampledWavelengths &) const noexcept override { return pmf(it); }
    [[nodiscard]] LightSampler::Selection select(
        Sampler::Instance &sampler, const Interaction &it,
        const SampledWavelengths &) const noexcept override {
        using namespace luisa::compute;
        auto u = sampler.generate_1d();
        auto n = static_cast<uint>(pipeline().lights().size());
        auto i = clamp(cast<uint>(u * static_cast<float>(n)), 0u, n - 1u);
        auto handle = pipeline().buffer<Light::Handle>(_light_buffer_id).read(i);
        return {handle.instance_id, handle.light_tag, pmf(it)};
    }
};

unique_ptr<LightSampler::Instance> UniformLightSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<UniformLightSamplerInstance>(
        this, pipeline, command_buffer);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::UniformLightSampler)

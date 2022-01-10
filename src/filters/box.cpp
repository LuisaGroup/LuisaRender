//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <scene/filter.h>

namespace luisa::render {

class BoxFilter final : public Filter {

public:
    BoxFilter(Scene *scene, const SceneNodeDesc *desc) noexcept : Filter{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "box"; }
};

class BoxFilterInstance final : public Filter::Instance {

public:
    explicit BoxFilterInstance(const BoxFilter *filter) noexcept : Filter::Instance{filter} {}
    Filter::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto u = sampler.generate_2d();
        return {.offset = u * node()->radius(), .weight = make_float3(1.0f)};
    }
};

luisa::unique_ptr<Filter::Instance> BoxFilter::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<BoxFilterInstance>(this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::BoxFilter)

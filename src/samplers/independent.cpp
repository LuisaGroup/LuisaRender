//
// Created by Mike on 2022/1/7.
//

#include <scene/sampler.h>

namespace luisa::render {

class IndependentSampler final : public Sampler {

public:
    IndependentSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Sampler{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Stream &stream, Pipeline &pipeline, uint2 resolution, uint spp) const noexcept override {
        return nullptr;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "independent"; }
};

}

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::IndependentSampler)

//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <scene/light_distribution.h>

namespace luisa::render {

class UniformLightDistribution final : public LightDistribution {

public:
    UniformLightDistribution(Scene *scene, const SceneNodeDesc *desc) noexcept : LightDistribution{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return "uniform"; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::UniformLightDistribution)

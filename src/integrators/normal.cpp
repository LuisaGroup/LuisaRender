//
// Created by Mike on 2022/1/7.
//

#include <scene/integrator.h>

namespace luisa::render {

class NormalVisualizer final : public Integrator {

public:
    NormalVisualizer(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "normal"; }
};

}

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalVisualizer)

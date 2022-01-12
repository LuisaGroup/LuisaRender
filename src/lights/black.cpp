//
// Created by Mike Smith on 2022/1/12.
//

#include <scene/light.h>

namespace luisa::render {

struct BlackLight final : public Light {
    BlackLight(Scene *scene, const SceneNodeDesc *desc) noexcept : Light{scene, desc} {}
    [[nodiscard]] bool is_black() const noexcept override { return true; }
    [[nodiscard]] bool is_virtual() const noexcept override { return true; }
    [[nodiscard]] string_view impl_type() const noexcept override { return "black"; }
    [[nodiscard]] uint encode(Pipeline &pipeline, CommandBuffer &command_buffer, uint instance_id, const Shape *shape) const noexcept override { return ~0u; }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(const Pipeline &pipeline, const Interaction &it) const noexcept override { return nullptr; }
};

}

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::BlackLight)

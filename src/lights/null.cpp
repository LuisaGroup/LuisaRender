//
// Created by Mike Smith on 2022/1/12.
//

#include <base/light.h>

namespace luisa::render {

struct NullLight final : public Light {
    NullLight(Scene *scene, const SceneNodeDesc *desc) noexcept : Light{scene, desc} {}
    [[nodiscard]] bool is_null() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return nullptr;
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NullLight)

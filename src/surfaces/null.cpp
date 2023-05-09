//
// Created by Mike Smith on 2022/1/12.
//

#include <base/surface.h>

namespace luisa::render {

struct NullSurface final : public Surface {

public:
    NullSurface(Scene *scene, const SceneNodeDesc *desc) noexcept : Surface{scene, desc} {}
    [[nodiscard]] bool is_null() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::string closure_identifier() const noexcept override { return luisa::string(impl_type()); }
    [[nodiscard]] uint properties() const noexcept override { return 0u; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        LUISA_ERROR_WITH_LOCATION("NullSurface cannot be instantiated.");
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NullSurface)

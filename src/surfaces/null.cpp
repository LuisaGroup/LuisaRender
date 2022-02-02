//
// Created by Mike Smith on 2022/1/12.
//

#include <base/surface.h>

namespace luisa::render {

struct NullSurface final : public Surface {
    NullSurface(Scene *scene, const SceneNodeDesc *desc) noexcept : Surface{scene, desc} {}
    [[nodiscard]] bool is_null() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint encode(Pipeline &, CommandBuffer &, uint, const Shape *) const noexcept override { return ~0u; }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &, const Interaction &,
        const SampledWavelengths &, Expr<float>) const noexcept override {
        LUISA_ERROR_WITH_LOCATION("NullSurface::decode() should never be called.");
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NullSurface)

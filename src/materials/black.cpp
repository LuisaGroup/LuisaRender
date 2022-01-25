//
// Created by Mike Smith on 2022/1/12.
//

#include <base/material.h>

namespace luisa::render {

struct BlackMaterial final : public Material {
    BlackMaterial(Scene *scene, const SceneNodeDesc *desc) noexcept : Material{scene, desc} {}
    [[nodiscard]] bool is_black() const noexcept override { return true; }
    [[nodiscard]] string_view impl_type() const noexcept override { return "black"; }
    [[nodiscard]] uint encode(Pipeline &, CommandBuffer &, uint, const Shape *) const noexcept override { return ~0u; }
    [[nodiscard]] unique_ptr<Closure> decode(const Pipeline &, const Interaction &, const SampledWavelengths &, Expr<float>) const noexcept override { return nullptr; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::BlackMaterial)

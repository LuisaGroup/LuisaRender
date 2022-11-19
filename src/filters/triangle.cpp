//
// Created by Mike Smith on 2022/1/16.
//

#include <base/filter.h>

namespace luisa::render {

struct TriangleFilter final : public Filter {
    TriangleFilter(Scene *scene, const SceneNodeDesc *desc) noexcept : Filter{scene, desc} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] float evaluate(float x) const noexcept override { return std::max(1.0f - std::abs(x / radius()), 0.0f); }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::TriangleFilter)

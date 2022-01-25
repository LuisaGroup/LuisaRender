//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <base/filter.h>

namespace luisa::render {

struct BoxFilter final : public Filter {
    BoxFilter(Scene *scene, const SceneNodeDesc *desc) noexcept : Filter{scene, desc} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "box"; }
    [[nodiscard]] float evaluate(float x) const noexcept override { return 1.0f; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::BoxFilter)

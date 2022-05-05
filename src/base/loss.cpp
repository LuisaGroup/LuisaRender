//
// Created by ChenXin on 2022/5/4.
//

#include <base/loss.h>

namespace luisa::render {

Loss::Loss(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LOSS} {}

Interaction pixel_xy2uv(Expr<uint2> pixel_id, uint2 resolution) noexcept {
    return Interaction{
        make_float3(),
        Float2{
            (pixel_id.x + 0.5f) / resolution.x,
            (pixel_id.y + 0.5f) / resolution.y}};
}

}// namespace luisa::render
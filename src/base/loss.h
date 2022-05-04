//
// Created by ChenXin on 2022/4/11.
//

#pragma once

#include <luisa-compute.h>

#include <base/camera.h>
#include <base/interaction.h>
#include <base/scene_node.h>
#include <base/spectrum.h>

namespace luisa::render {

using namespace luisa::compute;

class Loss : public SceneNode {

public:
    [[nodiscard]] static Interaction pixel_xy2uv(Expr<uint2> pixel_id, uint2 resolution) noexcept {
        return Interaction{
            make_float3(),
            Float2{
                (pixel_id.x + 0.5f) / resolution.x,
                (pixel_id.y + 0.5f) / resolution.y}};
    }

public:
    Loss(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual Float3 loss(const Camera::Instance *camera) const noexcept = 0;
    [[nodiscard]] virtual Float3 d_loss(const Camera::Instance *camera, Expr<uint2> pixel_id) const noexcept = 0;
};

}// namespace luisa::render
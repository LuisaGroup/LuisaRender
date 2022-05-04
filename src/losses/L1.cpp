//
// Created by ChenXin on 2022/5/4.
//

#include "base/loss.h"

namespace luisa::render {

using namespace luisa::compute;

class L1 final : public Loss {

public:
    L1(Scene *scene, const SceneNodeDesc *desc)
    noexcept
        : Loss{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

    [[nodiscard]] Float3 loss(const Camera::Instance *camera) const noexcept override {
        auto resolution = camera->node()->film()->resolution();
        auto loss_sum = def(make_float3(0.f));

        for (auto x = 0u; x < resolution.x; ++x) {
            for (auto y = 0u; y < resolution.y; ++y) {
                auto pixel_id = make_uint2(x, y);
                auto pixel_uv_it = pixel_xy2uv(pixel_id, resolution);

                auto rendered = camera->film()->read(pixel_id).average;
                auto target = camera->target()->evaluate(pixel_uv_it, 0.f).xyz();

                loss_sum += abs(rendered - target);
            }
        }

        return loss_sum;
    }

    [[nodiscard]] Float3 d_loss(const Camera::Instance *camera, Expr<uint2> pixel_id) const noexcept override {
        auto resolution = camera->node()->film()->resolution();
        auto pixel_uv_it = pixel_xy2uv(pixel_id, resolution);

        auto rendered = camera->film()->read(pixel_id).average;
        auto target = camera->target()->evaluate(pixel_uv_it, 0.f).xyz();

        return ite(rendered >= target, 1.0f, -1.0f);
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::L1)
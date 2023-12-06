//
// Created by ChenXin on 2022/5/4.
//

#include "base/loss.h"

namespace luisa::render {

using namespace luisa::compute;

class L2 final : public Loss {

public:
    L2(Scene *scene, const SceneNodeDesc *desc) noexcept : Loss{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Loss::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class L2Instance final : public Loss::Instance {

public:
    L2Instance(Pipeline &pipeline, const L2 *loss) noexcept
        : Loss::Instance{pipeline, loss} {}

    [[nodiscard]] Float3 loss(const Camera::Instance *camera, const SampledWavelengths &swl) const noexcept override;
    [[nodiscard]] Float3 d_loss(const Camera::Instance *camera, Expr<uint2> pixel_id,
                                const SampledWavelengths &swl) const noexcept override;
};

luisa::unique_ptr<Loss::Instance> L2::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<L2Instance>(pipeline, this);
}

Float3 L2Instance::loss(const Camera::Instance *camera, const SampledWavelengths &swl) const noexcept {
    auto resolution = camera->node()->film()->resolution();
    auto loss_sum = def(make_float3(0.f));

    for (auto x = 0u; x < resolution.x; ++x) {
        for (auto y = 0u; y < resolution.y; ++y) {
            auto pixel_id = make_uint2(x, y);
            auto pixel_uv_it = pixel_xy2uv(pixel_id, resolution);

            auto rendered = camera->film()->read(pixel_id).average;
            auto target = camera->target()->evaluate(pixel_uv_it, swl, 0.f).xyz();

            loss_sum += sqr(rendered - target);
        }
    }

    return loss_sum / float(resolution.x * resolution.y);
}

Float3 L2Instance::d_loss(const Camera::Instance *camera, Expr<uint2> pixel_id,
                          const SampledWavelengths &swl) const noexcept {
    auto resolution = camera->node()->film()->resolution();
    auto pixel_uv_it = pixel_xy2uv(pixel_id, resolution);

    auto rendered = camera->film()->read(pixel_id).average;
    auto target = camera->target()->evaluate(pixel_uv_it, swl, 0.f).xyz();

    return 2.f * (rendered - target);
    // return 2.f * (rendered - target) / float(resolution.x * resolution.y);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::L2)

//
// Created by Mike on 2022/1/7.
//

#include <util/rng.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

class NormalVisualizer final : public ProgressiveIntegrator {

private:
    bool _remap;
    bool _shading;

public:
    NormalVisualizer(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _remap{desc->property_bool_or_default("remap", true)},
          _shading{desc->property_bool_or_default("shading", true)} {}
    [[nodiscard]] auto remap() const noexcept { return _remap; }
    [[nodiscard]] auto shading() const noexcept { return _shading; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &cb) const noexcept override;
};

class NormalVisualizerInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto cs = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto swl = pipeline().spectrum()->sample(sampler()->generate_1d());
        auto path_weight = cs.weight;
        auto it = pipeline().geometry()->intersect(cs.ray);
        auto ns = def(make_float3(0.f));
        auto wo = -cs.ray->direction();
        $if(it->valid()) {
            if (node<NormalVisualizer>()->shading()) {
                $if(it->shape().has_surface()) {
                    pipeline().surfaces().dispatch(it->shape().surface_tag(), [&](auto surface) noexcept {
                        auto closure = surface->closure(it, swl, 1.f, time);
                        ns = closure->it()->shading().n();
                    });
                }
                $else {
                    ns = it->shading().n();
                };
            } else {
                ns = it->ng();
            }
            if (node<NormalVisualizer>()->remap()) {
                ns = ns * .5f + .5f;
            }
        };
        return path_weight * ns;
    }
};

luisa::unique_ptr<Integrator::Instance> NormalVisualizer::build(
    Pipeline &pipeline, CommandBuffer &cb) const noexcept {
    return luisa::make_unique<NormalVisualizerInstance>(pipeline, cb, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalVisualizer)

//
// Created by Mike Smith on 2022/1/10.
//

#include <util/imageio.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

using namespace compute;

class DirectLighting final : public ProgressiveIntegrator {

public:
    enum struct ImportanceSampling {
        LIGHT,
        SURFACE,
        BOTH
    };

private:
    ImportanceSampling _importance_sampling{};

public:
    DirectLighting(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc} {
        auto is = desc->property_string_or_default("importance_sampling", "both");
        for (auto &c : is) { c = static_cast<char>(tolower(c)); }
        if (is == "light") {
            _importance_sampling = ImportanceSampling::LIGHT;
        } else if (is == "material" || is == "surface" || is == "bsdf") {
            _importance_sampling = ImportanceSampling::SURFACE;
        } else {
            if (is != "both" && is != "mis" && is != "multiple") {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown importance sampling method \"{}\". Using \"both\" instead.",
                    is);
            }
            _importance_sampling = ImportanceSampling::BOTH;
        }
    }
    [[nodiscard]] auto importance_sampling() const noexcept { return _importance_sampling; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class DirectLightingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer,
                            Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        Instance::_render_one_camera(command_buffer, camera);
    }

    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto cs = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum Li{swl.dimension(), 0.f};

        auto importance_sampling = node<DirectLighting>()->importance_sampling();
        auto samples_lights = importance_sampling == DirectLighting::ImportanceSampling::LIGHT ||
                              importance_sampling == DirectLighting::ImportanceSampling::BOTH;
        auto samples_surfaces = importance_sampling == DirectLighting::ImportanceSampling::SURFACE ||
                                importance_sampling == DirectLighting::ImportanceSampling::BOTH;

        auto ray = cs.ray;

        $loop {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += cs.weight * eval.L;
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += cs.weight * eval.L;
                };
            }

            // compute direct lighting
            $if(!it->shape().has_surface()) { $break; };

            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            auto occluded = def(false);

            if (samples_lights) {
                // sample one light
                auto u_light_selection = sampler()->generate_1d();
                auto u_light_surface = sampler()->generate_2d();
                light_sample = light_sampler()->sample(
                    *it, u_light_selection, u_light_surface, swl, time);

                // trace shadow ray
                $if(light_sample.eval.pdf > 0.f &
                    light_sample.eval.L.any([](auto x) { return x > 0.f; })) {
                    occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
                };
            }

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = def(make_float2());
            if (samples_surfaces) { u_bsdf = sampler()->generate_2d(); }
            auto surface_sample = Surface::Sample::zero(swl.dimension());
            auto alpha_skip = def(false);
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(it, swl, wo, 1.f, time);

                // apply opacity map
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                }
                $else {
                    // some preparations
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    if (samples_lights) {
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.shadow_ray->direction();
                            auto eval = closure->evaluate(wo, wi);
                            $if(eval.pdf > 0.f) {
                                auto w = def(1.f);
                                // MIS if sampling surfaces as well
                                if (samples_surfaces) { w = balance_heuristic(light_sample.eval.pdf, eval.pdf); }
                                Li += w * cs.weight * eval.f * light_sample.eval.L / light_sample.eval.pdf;
                            };
                        };
                    }

                    // sample material
                    if (samples_surfaces) {
                        surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                        ray = it->spawn_ray(surface_sample.wi);
                    }
                };
            });

            $if(!alpha_skip) {
                if (samples_surfaces) {
                    // trace
                    auto bsdf_it = pipeline().geometry()->intersect(ray);

                    // miss
                    auto light_eval = Light::Evaluation::zero(swl.dimension());
                    $if(!bsdf_it->valid()) {
                        if (pipeline().environment()) {
                            light_eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        }
                    }
                    $else {
                        // hit light
                        if (!pipeline().lights().empty()) {
                            $if(bsdf_it->shape().has_light()) {
                                light_eval = light_sampler()->evaluate_hit(*bsdf_it, ray->origin(), swl, time);
                            };
                        }
                    };

                    $if(light_eval.pdf > 0.f & surface_sample.eval.pdf > 0.f) {
                        auto w = def(1.f);
                        if (samples_lights) { w = balance_heuristic(surface_sample.eval.pdf, light_eval.pdf); }
                        Li += cs.weight * w * surface_sample.eval.f * light_eval.L / surface_sample.eval.pdf;
                    };
                }
                $break;
            };
        };
        return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> DirectLighting::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<DirectLightingInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DirectLighting)

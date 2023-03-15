//
// Created by ChenXin on 2023/2/13.
//

#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <util/medium_tracker.h>
#include <util/rng.h>

namespace luisa::render {

using namespace compute;

class MegakernelVolumePathTracing final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    MegakernelVolumePathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 20u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelVolumePathTracingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        Instance::_render_one_camera(command_buffer, camera);
    }

    [[nodiscard]] static auto event(const Interaction *it, Expr<float3> wo, Expr<float3> wi) noexcept {
        const auto &shading = it->shading();
        auto wo_local = shading.world_to_local(wo);
        auto wi_local = shading.world_to_local(wi);
        return ite(
            wo_local.z * wi_local.z > 0.f,
            Surface::event_reflect,
            ite(
                wi_local.z > 0.f,
                Surface::event_exit,
                Surface::event_enter));
    }

    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension(), 0.f};
        SampledSpectrum r_u{swl.dimension(), 1.f}, r_l{swl.dimension(), 1.f};
        auto rr_depth = node<MegakernelVolumePathTracing>()->rr_depth();
        auto medium_tracker = MediumTracker(pipeline().printer());

        // initialize medium tracker
        auto env_medium_tag = pipeline().environment_medium_tag();
        pipeline().media().dispatch(env_medium_tag, [&](auto medium) {
            medium_tracker.enter(medium->priority(), def<MediumInfo>(env_medium_tag));
        });
        auto ray = camera_ray;
        $while(true) {
            auto it = pipeline().geometry()->intersect(ray);
            $if((!it->valid()) | (!it->shape()->has_surface())) { $break; };

            $if(it->shape()->has_medium()) {
                auto surface_tag = it->shape()->surface_tag();
                auto medium_tag = it->shape()->medium_tag();

                auto medium_info = def<MediumInfo>();
                medium_info.medium_tag = medium_tag;
                auto medium_priority = def<uint>(0u);

                pipeline().media().dispatch(medium_tag, [&](auto medium) {
                    medium_priority = medium->priority();
                });
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {
                    auto surface_event = event(it.get(), -ray->direction(), ray->direction());
                    $if(TEST_COND) {
                        pipeline().printer().verbose_with_location("surface event={}", surface_event);
                    };
                    // update medium tracker
                    $switch(surface_event) {
                        $case(Surface::event_enter) {
                            medium_tracker.enter(medium_priority, medium_info);
                            $if(TEST_COND) {
                                pipeline().printer().verbose_with_location("enter: priority={}, medium_tag={}", medium_priority, medium_tag);
                            };
                        };
                        $case(Surface::event_exit) {
                            $if(medium_tracker.exist(medium_priority, medium_info)) {
                                medium_tracker.exit(medium_priority, medium_info);
                                $if(TEST_COND) {
                                    pipeline().printer().verbose_with_location("exit exist: priority={}, medium_tag={}", medium_priority, medium_tag);
                                };
                            }
                            $else {
                                medium_tracker.enter(medium_priority, medium_info);
                                $if(TEST_COND) {
                                    pipeline().printer().verbose_with_location("exit nonexistent: priority={}, medium_tag={}", medium_priority, medium_tag);
                                };
                            };
                        };
                    };
                });
            };
            $if(TEST_COND) {
                pipeline().printer().verbose_with_location("it->shape()->has_medium()={}", it->shape()->has_medium());
                pipeline().printer().verbose_with_location("medium tracker size={}", medium_tracker.size());
                pipeline().printer().verbose_with_location("it->p()=({}, {}, {})", it->p().x, it->p().y, it->p().z);
                pipeline().printer().verbose("");
            };
            ray = it->spawn_ray(ray->direction());
        };
        $if(TEST_COND) {
            pipeline().printer().verbose_with_location("Final medium tracker size={}", medium_tracker.size());
        };

        ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        auto depth = def(0u);
        $while(depth < node<MegakernelVolumePathTracing>()->max_depth()) {
            auto eta_scale = def(1.f);
            auto u_rr = def(0.f);
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // trace
            auto it = pipeline().geometry()->intersect(ray);
            auto has_medium = it->shape()->has_medium();

            // sample the participating medium
            auto is_scattered = def(false);
            $if(!medium_tracker.vacuum()) {
                // Normalize ray direction and update t_max accordingly
                ray->set_direction(normalize(ray->direction()));
                auto t_max = ite(it->valid(), length(it->p() - ray->origin()), Interaction::default_t_max);

                // Initialize RNG for sampling the majorant transmittance
                auto hash0 = U64(as<UInt2>(sampler()->generate_2d()));
                auto hash1 = U64(as<UInt2>(sampler()->generate_2d()));
                PCG32 rng(hash0, hash1);

                // Sample medium using delta tracking
                Float u = sampler()->generate_1d();
                Float u_behavior = sampler()->generate_1d();

                auto medium_tag = medium_tracker.current().medium_tag;
                pipeline().media().dispatch(medium_tag, [&](auto medium) {
                    auto closure = medium->closure(ray, swl, time);

                    closure->sampleT_maj(
                        t_max, u, rng,
                        [&](luisa::unique_ptr<Medium::Closure> closure, Expr<float3> p,
                            SampledSpectrum sigma_maj, SampledSpectrum T_maj) {

                        });
                });
            };

            // sample the surface
            $if(!is_scattered) {
                // miss
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    }
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape()->has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                }

                $if(!it->shape()->has_surface()) { $break; };

                // generate uniform samples
                auto u_light_selection = sampler()->generate_1d();
                auto u_light_surface = sampler()->generate_2d();
                auto u_lobe = sampler()->generate_1d();
                auto u_bsdf = sampler()->generate_2d();

                // sample one light
                auto light_sample = light_sampler()->sample(
                    *it, u_light_selection, u_light_surface, swl, time);

                // trace shadow ray
                auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

                auto medium_tag = it->shape()->medium_tag();
                auto medium_priority = def(0u);
                auto medium_info = def<MediumInfo>();
                medium_info.medium_tag = medium_tag;
                auto eta = def(1.f);
                $if(!medium_tracker.vacuum()) {
                    pipeline().media().dispatch(medium_tracker.current().medium_tag, [&](auto medium) {
                        auto closure = medium->closure(ray, swl, time);
                        eta = closure->eta();
                    });
                };
                $if(has_medium) {
                    pipeline().media().dispatch(medium_tag, [&](auto medium) {
                        medium_priority = medium->priority();
                        auto closure = medium->closure(ray, swl, time);
                        $if(TEST_COND) {
                            pipeline().printer().verbose_with_location("eta={}", closure->eta());
                        };
                    });
                };

                // evaluate material
                auto surface_tag = it->shape()->surface_tag();
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    // create closure
                    auto closure = surface->closure(it, swl, eta, time);

                    // apply opacity map
                    auto alpha_skip = def(false);
                    if (auto o = closure->opacity()) {
                        auto opacity = saturate(*o);
                        alpha_skip = u_lobe >= opacity;
                        u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                    }

                    $if(alpha_skip) {
                        ray = it->spawn_ray(ray->direction());
                        pdf_bsdf = 1e16f;
                    }
                    $else {
                        if (auto dispersive = closure->is_dispersive()) {
                            $if(*dispersive) { swl.terminate_secondary(); };
                        }
                        // direct lighting
                        auto wo = -ray->direction();
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.shadow_ray->direction();
                            auto eval = closure->evaluate(wo, wi);
                            auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                     light_sample.eval.pdf;
                            Li += w * beta * eval.f * light_sample.eval.L;
                        };
                        // sample material
                        auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);

                        ray = it->spawn_ray(surface_sample.wi);
                        pdf_bsdf = surface_sample.eval.pdf;
                        auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                        beta *= w * surface_sample.eval.f;
                        // apply eta scale & update medium tracker
                        auto eta = closure->eta().value_or(1.f);
                        $if(has_medium) {
                            $switch(surface_sample.event) {
                                $case(Surface::event_enter) {
                                    eta_scale = sqr(eta);
                                    medium_tracker.enter(medium_priority, medium_info);
                                };
                                $case(Surface::event_exit) {
                                    eta_scale = sqr(1.f / eta);
                                    medium_tracker.exit(medium_priority, medium_info);
                                };
                            };
                        };
                    };
                });
            };

            beta = zero_if_any_nan(beta);
            $if(TEST_COND) {
                pipeline().printer().verbose_with_location("beta_before_break=({}, {}, {})", beta[0u], beta[1u], beta[2u]);
            };
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<MegakernelVolumePathTracing>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
            depth += 1u;

            $if(TEST_COND) {
                pipeline().printer().verbose_with_location("it->p(): ({}, {}, {})", it->p().x, it->p().y, it->p().z);
                pipeline().printer().verbose_with_location(
                    "depth={}, is_scattered={}, beta=({}, {}, {}), pdf_bsdf={}, Li: ({}, {}, {})",
                    depth, is_scattered, beta[0u], beta[1u], beta[2u], pdf_bsdf, Li[0u], Li[1u], Li[2u]);
                pipeline().printer().verbose("");
            };
        };
        return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> MegakernelVolumePathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelVolumePathTracingInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelVolumePathTracing)

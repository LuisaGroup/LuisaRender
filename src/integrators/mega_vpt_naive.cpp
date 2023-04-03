//
// Created by ChenXin on 2023/3/30.
//

#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <util/medium_tracker.h>
#include <base/medium.h>
#include <base/phase_function.h>
#include <util/rng.h>

namespace luisa::render {

using namespace compute;

class MegakernelVolumePathTracingNaive final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    MegakernelVolumePathTracingNaive(Scene *scene, const SceneNodeDesc *desc) noexcept
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

class MegakernelVolumePathTracingNaiveInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

    struct Evaluation {
        SampledSpectrum f;
        Float pdf;
        [[nodiscard]] static auto one(uint spec_dim) noexcept {
            return Evaluation{
                .f = SampledSpectrum{spec_dim, 1.f},
                .pdf = 1.f};
        }
    };

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        Instance::_render_one_camera(command_buffer, camera);
    }

    [[nodiscard]] UInt _event(const SampledWavelengths &swl, luisa::shared_ptr<Interaction> it, Expr<float> time,
                              Expr<float3> wo, Expr<float3> wi) const noexcept {
        Float3 wo_local, wi_local;
        $if(it->shape().has_surface()) {
            pipeline().surfaces().dispatch(it->shape().surface_tag(), [&](auto surface) noexcept {
                auto closure = surface->closure(it, swl, wo, 1.f, time);
                auto shading = closure->it()->shading();
                wo_local = shading.world_to_local(wo);
                wi_local = shading.world_to_local(wi);
            });
        }
        $else {
            auto shading = it->shading();
            wo_local = shading.world_to_local(wo);
            wi_local = shading.world_to_local(wi);
        };
        return ite(
            wo_local.z * wi_local.z > 0.f,
            Surface::event_reflect,
            ite(
                wi_local.z > 0.f,
                Surface::event_exit,
                Surface::event_enter));
    }

    [[nodiscard]] Evaluation _transmittance(
        Expr<uint> frame_index, Expr<uint2> pixel_id, Expr<float> time, const SampledWavelengths &swl, PCG32 &rng,
        MediumTracker medium_tracker, Var<Ray> origin_ray) const noexcept {
        auto t_max = origin_ray->t_max();
        auto dir = origin_ray->direction();
        auto ray = def(origin_ray);
        auto light_p = origin_ray->origin() + dir * t_max;
        auto transmittance = Evaluation::one(swl.dimension());
        transmittance.pdf = 0.f;

        // trace shadow ray
        $while(any(transmittance.f > 0.f)) {
            auto it = pipeline().geometry()->intersect(ray);

            // end tracing
            $if(!it->valid()) { $break; };

            auto t2surface = length(it->p() - ray->origin());
            auto has_medium = it->shape().has_medium();
            auto medium_tag = it->shape().medium_tag();
            auto medium_priority = def(Medium::VACUUM_PRIORITY);
            auto wo = -dir;
            auto wi = dir;
            auto surface_event = _event(swl, it, time, wo, wi);

            // transmittance through medium
            $if(!medium_tracker.vacuum()) {
                pipeline().media().dispatch(medium_tracker.current().medium_tag, [&](auto medium) {
                    medium_priority = medium->priority();
                    auto closure = medium->closure(ray, swl, time);
                    auto medium_evaluation = closure->transmittance(t2surface, rng);
                    transmittance.f *= medium_evaluation.f;
                    transmittance.pdf += medium_evaluation.pdf;
                });
            };

            // update medium tracker
            $if(has_medium) {
                pipeline().media().dispatch(medium_tag, [&](auto medium) {
                    medium_priority = medium->priority();
                });
                auto medium_info = make_medium_info(medium_priority, medium_tag);
                $if(surface_event == Surface::event_exit) {
                    medium_tracker.exit(medium_priority, medium_info);
                }
                $else {
                    medium_tracker.enter(medium_priority, medium_info);
                };
            };

            // hit solid/transmissive surface
            $if(it->shape().has_surface()) {
                auto surface_tag = it->shape().surface_tag();
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {
                    auto closure = surface->closure(it, swl, wo, 1.f, time);
                    // alpha skip
                    if (auto o = closure->opacity()) {
                        auto opacity = saturate(*o);
                        auto completely_opaque = opacity == 1.f;
                        transmittance.f = ite(completely_opaque, 0.f, transmittance.f);
                        transmittance.pdf = ite(completely_opaque, 1e16f, transmittance.pdf + 1.f / (1.f - opacity));
                    }
                    // surface transmit
                    else {
                        auto surface_evaluation = closure->evaluate(wo, wi);
                        transmittance.f *= surface_evaluation.f;
                        transmittance.pdf += surface_evaluation.pdf;
                    }
                });
            };

            ray = it->spawn_ray_to(light_p);

            $if(TEST_COND) {
                pipeline().printer().verbose_with_location(
                    "transmittance: f=({}, {}, {}), pdf={}",
                    transmittance.f[0u], transmittance.f[1u], transmittance.f[2u], transmittance.pdf);
            };
        };

        return transmittance;
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
        auto rr_depth = node<MegakernelVolumePathTracingNaive>()->rr_depth();
        auto medium_tracker = MediumTracker(pipeline().printer());

        // Initialize RNG for sampling the majorant transmittance
        auto hash0 = U64(as<UInt2>(sampler()->generate_2d()));
        auto hash1 = U64(as<UInt2>(sampler()->generate_2d()));
        PCG32 rng(hash0, hash1);

        // initialize medium tracker
        auto env_medium_tag = pipeline().environment_medium_tag();
        pipeline().media().dispatch(env_medium_tag, [&](auto medium) {
            medium_tracker.enter(medium->priority(), make_medium_info(medium->priority(), env_medium_tag));
        });
        auto ray = camera_ray;
        // TODO: bug in initialization of medium tracker where the angle between shared edge is small
        auto depth_track = def<uint>(0u);
        $while(true) {
            auto it = pipeline().geometry()->intersect(ray);
            $if(!it->valid()) { $break; };

            $if(TEST_COND) {
                pipeline().printer().verbose_with_location("depth={}", depth_track);
            };

            $if(it->shape().has_medium()) {
                auto surface_tag = it->shape().surface_tag();
                auto medium_tag = it->shape().medium_tag();

                auto medium_priority = def(Medium::VACUUM_PRIORITY);
                pipeline().media().dispatch(medium_tag, [&](auto medium) {
                    medium_priority = medium->priority();
                });
                auto medium_info = make_medium_info(medium_priority, medium_tag);

                // deal with medium tracker
                auto surface_event = _event(swl, it, time, -ray->direction(), ray->direction());
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {
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
                pipeline().printer().verbose_with_location("medium tracker size={}", medium_tracker.size());
                auto dir = ray->direction();
                auto origin = ray->origin();
                pipeline().printer().verbose_with_location("ray->origin()=({}, {}, {})", origin.x, origin.y, origin.z);
                pipeline().printer().verbose_with_location("ray->direction()=({}, {}, {})", dir.x, dir.y, dir.z);
                pipeline().printer().verbose_with_location("it->p()=({}, {}, {})", it->p().x, it->p().y, it->p().z);
                pipeline().printer().verbose_with_location("it->shape().has_medium()={}", it->shape().has_medium());
                pipeline().printer().verbose("");
            };
            ray = it->spawn_ray(ray->direction());
            depth_track += 1u;
        };
        $if(TEST_COND) {
            pipeline().printer().verbose_with_location("Final medium tracker size={}", medium_tracker.size());
            pipeline().printer().verbose("");
        };

        ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        auto eta_scale = def(1.f);
        auto max_depth = node<MegakernelVolumePathTracingNaive>()->max_depth();
        $for(depth, max_depth) {
            auto eta = def(1.f);
            auto u_rr = def(0.f);
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // trace
            auto it = pipeline().geometry()->intersect(ray);
            auto has_medium = it->shape().has_medium();
            auto t_max = ite(it->valid(), length(it->p() - ray->origin()), Interaction::default_t_max);

            $if(TEST_COND) {
                pipeline().printer().verbose_with_location("depth={}", depth);
                pipeline().printer().verbose_with_location("before: medium tracker size={}, priority={}, tag={}",
                                                           medium_tracker.size(), medium_tracker.current().priority, medium_tracker.current().medium_tag);
                pipeline().printer().verbose_with_location(
                    "ray=({}, {}, {}) + t * ({}, {}, {})",
                    ray->origin().x, ray->origin().y, ray->origin().z,
                    ray->direction().x, ray->direction().y, ray->direction().z);
                pipeline().printer().verbose_with_location("it->p()=({}, {}, {})", it->p().x, it->p().y, it->p().z);
            };

            auto medium_sample = Medium::Sample::zero(swl.dimension());
            // sample the participating medium
            $if(!medium_tracker.vacuum()) {
                // direct light
                // generate uniform samples
                auto u_light_selection = sampler()->generate_1d();
                auto u_light_surface = sampler()->generate_2d();

                // sample one light
                auto it_medium = Interaction{ray->origin()};
                auto light_sample = light_sampler()->sample(
                    it_medium, u_light_selection, u_light_surface, swl, time);

                // trace shadow ray
                auto transmittance_evaluation = _transmittance(frame_index, pixel_id, time, swl, rng, medium_tracker, light_sample.shadow_ray);
                $if(transmittance_evaluation.pdf > 0.f) {
                    auto w = 1.f / (pdf_bsdf + transmittance_evaluation.pdf + light_sample.eval.pdf);
                    Li += w * beta * transmittance_evaluation.f * light_sample.eval.L;
                };

                auto medium_tag = medium_tracker.current().medium_tag;
                pipeline().media().dispatch(medium_tag, [&](auto medium) {
                    auto closure = medium->closure(ray, swl, time);
                    eta = closure->eta();

                    if (!closure->instance()->node()->is_vacuum()) {
                        medium_sample = closure->sample(t_max, rng);

                        // update ray
                        ray = medium_sample.ray;
                        auto w = ite(medium_sample.eval.pdf > 0.f, 1.f / medium_sample.eval.pdf, 0.f);
                        beta *= medium_sample.eval.f * w;
                        pdf_bsdf = medium_sample.eval.pdf;
                    }
                });
            };

            // sample the surface
            $if((medium_sample.medium_event == Medium::event_invalid) | (medium_sample.medium_event == Medium::event_hit_surface)) {
                // miss, environment light
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    }
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        $if(TEST_COND) {
                            pipeline().printer().verbose_with_location(
                                "hit light: "
                                "pdf_bsdf={},"
                                "eval.pdf={}, "
                                "balance_heuristic(pdf_bsdf, eval.pdf)={}, "
                                "eval.L=({}, {}, {}), "
                                "beta=({}, {}, {})",
                                pdf_bsdf,
                                eval.pdf,
                                balance_heuristic(pdf_bsdf, eval.pdf),
                                eval.L[0u], eval.L[1u], eval.L[2u],
                                beta[0u], beta[1u], beta[2u]);
                        };
                    };
                }

                // hit ordinary surface
                $if(!it->shape().has_surface()) { $break; };

                // generate uniform samples
                auto u_light_selection = sampler()->generate_1d();
                auto u_light_surface = sampler()->generate_2d();
                auto u_lobe = sampler()->generate_1d();
                auto u_bsdf = sampler()->generate_2d();

                // sample one light
                auto light_sample = light_sampler()->sample(
                    *it, u_light_selection, u_light_surface, swl, time);

                // trace shadow ray
                //                    auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);
                auto transmittance_evaluation = _transmittance(frame_index, pixel_id, time, swl, rng, medium_tracker, light_sample.shadow_ray);

                auto medium_tag = it->shape().medium_tag();
                auto medium_priority = def(Medium::VACUUM_PRIORITY);
                auto eta_next = def(1.f);
                $if(has_medium) {
                    pipeline().media().dispatch(medium_tag, [&](auto medium) {
                        auto closure = medium->closure(ray, swl, time);
                        medium_priority = medium->priority();
                        eta_next = closure->eta();
                        $if(TEST_COND) {
                            pipeline().printer().verbose_with_location("eta_next={}", eta_next);
                        };
                    });
                };
                auto medium_info = make_medium_info(medium_priority, medium_tag);
                medium_info.medium_tag = medium_tag;

                // evaluate material
                auto surface_tag = it->shape().surface_tag();
                auto surface_event_skip = _event(swl, it, time, -ray->direction(), ray->direction());
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    // create closure
                    auto wo = -ray->direction();
                    auto closure = surface->closure(it, swl, wo, eta, time);

                    // apply opacity map
                    auto alpha_skip = def(false);
                    if (auto o = closure->opacity()) {
                        auto opacity = saturate(*o);
                        alpha_skip = u_lobe >= opacity;
                        u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                    }

                    UInt surface_event;
                    $if(alpha_skip | !medium_tracker.true_hit(medium_info.medium_tag)) {
                        surface_event = surface_event_skip;
                        ray = it->spawn_ray(ray->direction());
                        pdf_bsdf = 1e16f;
                    }
                    $else {
                        if (auto dispersive = closure->is_dispersive()) {
                            $if(*dispersive) { swl.terminate_secondary(); };
                        }

                        // direct lighting
                        $if(light_sample.eval.pdf > 0.0f) {
                            auto wi = light_sample.shadow_ray->direction();
                            auto eval = closure->evaluate(wo, wi);
                            auto w = 1.f / (light_sample.eval.pdf + eval.pdf + transmittance_evaluation.pdf);
                            Li += w * beta * eval.f * light_sample.eval.L * transmittance_evaluation.f;
                            //                                auto w = 1.f / (light_sample.eval.pdf + eval.pdf);
                            //                                Li += w * beta * eval.f * light_sample.eval.L;
                            $if(TEST_COND) {
                                pipeline().printer().verbose_with_location(
                                    "direct lighting: "
                                    "eval.f=({}, {}, {}), "
                                    "eval.pdf={}, "
                                    "light_sample.eval.L=({}, {}, {}), "
                                    "light_sample.eval.pdf={},"
                                    "beta=({}, {}, {})",
                                    eval.f[0u], eval.f[1u], eval.f[2u],
                                    eval.pdf,
                                    light_sample.eval.L[0u], light_sample.eval.L[1u], light_sample.eval.L[2u],
                                    light_sample.eval.pdf,
                                    beta[0u], beta[1u], beta[2u]);
                            };
                        };

                        // sample material
                        auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                        surface_event = surface_sample.event;
                        auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);

                        pdf_bsdf = surface_sample.eval.pdf;
                        ray = it->spawn_ray(surface_sample.wi);
                        beta *= w * surface_sample.eval.f;

                        // apply eta scale & update medium tracker
                        $if(has_medium) {
                            $switch(surface_event) {
                                $case(Surface::event_enter) {
                                    eta_scale = sqr(eta_next / eta);
                                };
                                $case(Surface::event_exit) {
                                    eta_scale = sqr(eta / eta_next);
                                };
                            };
                        };
                    };

                    $if(has_medium) {
                        $switch(surface_event) {
                            $case(Surface::event_enter) {
                                medium_tracker.enter(medium_priority, medium_info);
                            };
                            $case(Surface::event_exit) {
                                medium_tracker.exit(medium_priority, medium_info);
                            };
                        };
                    };

                    $if(TEST_COND) {
                        pipeline().printer().verbose_with_location(
                            "surface event={}, priority={}, tag={}",
                            surface_event, medium_priority, medium_tag);
                    };
                });
            };

            $if(TEST_COND) {
                pipeline().printer().verbose_with_location(
                    "medium event={}, beta=({}, {}, {}), pdf_bsdf={}, Li=({}, {}, {})",
                    medium_sample.medium_event, beta[0u], beta[1u], beta[2u], pdf_bsdf, Li[0u], Li[1u], Li[2u]);
            };

            beta = zero_if_any_nan(beta);
            $if(all(beta <= 0.f)) { $break; };
            // rr
            auto rr_threshold = node<MegakernelVolumePathTracingNaive>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };

            $if(TEST_COND) {
                pipeline().printer().verbose_with_location("beta=({}, {}, {})", beta[0u], beta[1u], beta[2u]);
                pipeline().printer().verbose_with_location("after: medium tracker size={}, priority={}, tag={}",
                                                           medium_tracker.size(), medium_tracker.current().priority, medium_tracker.current().medium_tag);
                pipeline().printer().verbose("");
            };
        };
        return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> MegakernelVolumePathTracingNaive::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelVolumePathTracingNaiveInstance>(
        pipeline, command_buffer, this);
}

#undef TEST_COND

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelVolumePathTracingNaive)

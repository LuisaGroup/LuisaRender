//
// Created by ChenXin on 2022/2/23.
//
// #include <iostream>
#include "core/logging.h"
#include <luisa-compute.h>
#include <util/imageio.h>
#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <core/stl.h>

namespace luisa::render {

using namespace compute;

class MegakernelReplayDiff final : public DifferentiableIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    MegakernelReplayDiff(Scene *scene, const SceneNodeDesc *desc) noexcept
        : DifferentiableIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelReplayDiffInstance : public DifferentiableIntegrator::Instance {
public:
    using DifferentiableIntegrator::Instance::Instance;
    [[nodiscard]] virtual Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                                        Expr<uint2> pixel_id, Expr<float> time) const noexcept override;
    void _render_one_camera_backward(
        CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera, Buffer<float> &grad_in) noexcept override;
};

luisa::unique_ptr<Integrator::Instance> MegakernelReplayDiff::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelReplayDiffInstance>(pipeline, command_buffer, this);
}

Float3 MegakernelReplayDiffInstance::Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) const noexcept {

    sampler()->start(pixel_id, frame_index);
    auto u_filter = sampler()->generate_pixel_2d();
    auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
    auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
    auto spectrum = pipeline().spectrum();
    auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
    SampledSpectrum beta{swl.dimension(), camera_weight};
    SampledSpectrum Li{swl.dimension()};

    auto ray = camera_ray;
    auto pdf_bsdf = def(1e16f);
    $for(depth, node<MegakernelReplayDiff>()->max_depth()) {

        // trace
        auto wo = -ray->direction();
        auto it = pipeline().geometry()->intersect(ray);

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
            $outline {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                };
            };
        }

        $if(!it->shape().has_surface()) { $break; };

        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        auto u_lobe = sampler()->generate_1d();
        auto u_bsdf = sampler()->generate_2d();

        auto u_rr = def(0.f);
        auto rr_depth = node<MegakernelReplayDiff>()->rr_depth();
        $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

        // generate uniform samples
        auto light_sample = LightSampler::Sample::zero(swl.dimension());
        $outline {
            // sample one light
            light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);
        };

        // trace shadow ray
        auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

        // evaluate material
        auto surface_tag = it->shape().surface_tag();
        auto eta_scale = def(1.f);

        $outline {
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](const Surface::Closure *closure) noexcept {
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
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) / light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                    };
                    // sample material
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                };
            });
        };

        beta = zero_if_any_nan(beta);
        $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
        auto rr_threshold = node<MegakernelReplayDiff>()->rr_threshold();
        auto q = max(beta.max() * eta_scale, .05f);
        $if(depth + 1u >= rr_depth) {
            $if(q < rr_threshold & u_rr >= q) { $break; };
            beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
        };
    };
    return spectrum->srgb(swl, Li);
}

void MegakernelReplayDiffInstance::_render_one_camera_backward(
    CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera, Buffer<float> & grad_in) noexcept {

    //auto grad_in_real = grad_in.view().as<float4>();
    auto spp = camera->node()->spp();
    auto resolution = camera->node()->film()->resolution();
    camera->film_grad()->prepare(command_buffer);

    LUISA_INFO("Start backward propagation.");

    auto pt = this;

    auto sampler = pt->sampler();
    auto env = pipeline().environment();

    auto pixel_count = resolution.x * resolution.y;
    auto light_sampler = pt->light_sampler();
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << commit() << synchronize();
    auto pt_exact = pt->node<MegakernelReplayDiff>();

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
    auto pixel_checked = make_uint2(120u, 280u);
#endif

    auto render_1spp_shader_iter = _render_1spp_shaders.find(camera);
    if (render_1spp_shader_iter == _render_1spp_shaders.end()) {

        using namespace luisa::compute;

        Kernel2D render_kernel_1spp = [&](UInt frame_index, Float time, Float shutter_weight, ImageFloat Li_1spp) noexcept {
            set_block_size(16u, 16u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler->start(pixel_id, frame_index);
            auto u_filter = sampler->generate_pixel_2d();
            auto u_lens = camera->node()->requires_lens_sampling() ? sampler->generate_2d() : make_float2(.5f);
            auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
            auto spectrum = pipeline().spectrum();
            auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension()};

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            $if(all(pixel_id == pixel_checked)) {
                pipeline().printer().info("Li_1spp forward: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
            };
#endif

            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);
            $for(depth, pt->node<MegakernelReplayDiff>()->max_depth()) {

                // trace
                auto wo = -ray->direction();
                auto it = pipeline().geometry()->intersect(ray);

                // miss
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                        $if(all(pixel_id == pixel_checked)) {
                            pipeline().printer().info("miss and break: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                        };
#endif
                    }
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape().has_light()) {
                        auto eval = light_sampler->evaluate_hit(*it, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                        $if(all(pixel_id == pixel_checked)) {
                            pipeline().printer().info("hit light: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                        };
#endif
                    };
                }

                $if(!it->shape().has_surface()) { $break; };

                // sample one light
                auto u_light_selection = sampler->generate_1d();
                auto u_light_surface = sampler->generate_2d();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();

                auto u_rr = def(0.f);
                auto rr_depth = node<MegakernelReplayDiff>()->rr_depth();
                $if(depth + 1u >= rr_depth) { u_rr = sampler->generate_1d(); };

                auto light_sample = LightSampler::Sample::zero(swl.dimension());
                $outline {
                    // sample one light
                    light_sample = light_sampler->sample(
                        *it, u_light_selection, u_light_surface, swl, time);
                };

                // trace shadow ray
                auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

                // evaluate material
                auto surface_tag = it->shape().surface_tag();
                auto eta_scale = def(1.f);
                $outline {
                    PolymorphicCall<Surface::Closure> call;
                    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                        surface->closure(call, *it, swl, wo, 1.f, time);
                    });
                    call.execute([&](const Surface::Closure *closure) noexcept {
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
                            $if(light_sample.eval.pdf > 0.0f & !occluded) {
                                auto wi = light_sample.shadow_ray->direction();
                                auto eval = closure->evaluate(wo, wi);
                                auto mis_weight = balance_heuristic(light_sample.eval.pdf, eval.pdf) / light_sample.eval.pdf;
                                Li += mis_weight * beta * eval.f * light_sample.eval.L;

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                                $if(all(pixel_id == pixel_checked)) {
                                    pipeline().printer().info("direct lighted: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                                };
#endif
                            };

                            // sample material
                            auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                            ray = it->spawn_ray(surface_sample.wi);
                            pdf_bsdf = surface_sample.eval.pdf;
                            auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                            beta *= w * surface_sample.eval.f;

                            // apply eta scale
                            auto eta = closure->eta().value_or(1.f);
                            $switch(surface_sample.event) {
                                $case(Surface::event_enter) { eta_scale = sqr(eta); };
                                $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                            };
                        };
                    });
                };
                // rr
                beta = zero_if_any_nan(beta);
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto rr_threshold = node<MegakernelReplayDiff>()->rr_threshold();
                auto q = max(beta.max() * eta_scale, .05f);
                $if(depth + 1u >= rr_depth) {
                    $if(q < rr_threshold & u_rr >= q) { $break; };
                    beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
                };
            };
            Li_1spp.write(pixel_id, make_float4(spectrum->srgb(swl, Li * shutter_weight), 1.f));

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            $if(all(pixel_id == pixel_checked)) {
                pipeline().printer().info("done: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
            };
#endif
        };
        auto render_shader = pipeline().device().compile(render_kernel_1spp);
        render_1spp_shader_iter = _render_1spp_shaders.emplace(camera, std::move(render_shader)).first;
    }
    auto &&render_1spp_shader = render_1spp_shader_iter->second;

    auto bp_shader_iter = _bp_shaders.find(camera);
    if (bp_shader_iter == _bp_shaders.end()) {

        using namespace luisa::compute;
        Kernel2D bp_kernel = [&](UInt frame_index, Float time, Float shutter_weight, ImageFloat Li_1spp, BufferFloat grad_in_real) noexcept {
            set_block_size(16u, 16u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler->start(pixel_id, frame_index);
            auto u_filter = sampler->generate_pixel_2d();
            auto u_lens = camera->node()->requires_lens_sampling() ? sampler->generate_2d() : make_float2(.5f);
            auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
            auto spectrum = pipeline().spectrum();
            auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension(), 0.f};
            //Why set this grad weight?
            auto grad_weight = shutter_weight * static_cast<float>(pt->node<MegakernelReplayDiff>()->max_depth());//?

            // for rendering grad map
            SampledSpectrum grad_map{swl.dimension(), 0.f};

            auto Li_last_pass = Li_1spp.read(pixel_id);
            Li[0u] = Li_last_pass[0u];
            Li[1u] = Li_last_pass[1u];
            Li[2u] = Li_last_pass[2u];

            //            SampledSpectrum d_loss{swl.dimension(), float(pixel_count)};
            SampledSpectrum d_loss{swl.dimension(), 1.f};


            //auto d_loss_float3 = pt->loss()->d_loss(camera, pixel_id, swl);
            //auto  = Float3(1.0,1.0,1.0);
            auto resolution = camera->film()->node()->resolution();
            //auto d_loss_float3 = 
            //grad_in_real.offset(pixel_id).xyz();

            for (auto i = 0u; i < 3u; ++i) {
                d_loss[i] *= grad_in_real->read((pixel_id.y * resolution.x + pixel_id.x)*4+i);
            }
            //Float3 rendered = camera->film()->read(pixel_id).average;
            //auto pixel_uv_it = pixel_xy2uv(pixel_id, resolution);
            //Float3 target = camera->target()->evaluate(pixel_uv_it, swl, 0.f).xyz();

            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            $if(all(pixel_id == pixel_checked)) {
                pipeline().printer().info("Li_1spp backward start: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
            };
#endif

            $for(depth, pt->node<MegakernelReplayDiff>()->max_depth()) {

                // trace
                auto wo = -ray->direction();
                auto it = pipeline().geometry()->intersect(ray);

                // miss, environment light
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler->evaluate_miss(
                            ray->direction(), swl, time);
                        Li -= beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                        $if(all(pixel_id == pixel_checked)) {
                            pipeline().printer().info("miss and break: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                        };
#endif
                    }
                    // TODO : backward environment light
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape().has_light()) {
                        auto eval = light_sampler->evaluate_hit(
                            *it, ray->origin(), swl, time);
                        Li -= beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        auto dlight = d_loss * beta * balance_heuristic(pdf_bsdf, eval.pdf);
                        light_sampler->backward_hit(*it, ray->origin(), swl, time, dlight);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                        $if(all(pixel_id == pixel_checked)) {
                            pipeline().printer().info("after -hit: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                        };
#endif
                    };
                    // TODO : backward hit light
                }

                $if(!it->shape().has_surface()) { $break; };

                // sample one light
                auto u_light_selection = sampler->generate_1d();
                auto u_light_surface = sampler->generate_2d();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();
                auto u_rr = def(0.f);
                auto rr_depth = node<MegakernelReplayDiff>()->rr_depth();
                $if(depth + 1u >= rr_depth) { u_rr = sampler->generate_1d(); };

                auto light_sample = LightSampler::Sample::zero(swl.dimension());
                $outline {
                    // sample one light
                    light_sample = light_sampler->sample(
                        *it, u_light_selection, u_light_surface, swl, time);
                };

                // trace shadow ray
                auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

                // evaluate material
                auto surface_tag = it->shape().surface_tag();
                auto eta_scale = def(1.f);
                $outline {
                    PolymorphicCall<Surface::Closure> call;
                    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                        surface->closure(call, *it, swl, wo, 1.f, time);
                    });
                    call.execute([&](const Surface::Closure *closure) noexcept {
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
                            $if(light_sample.eval.pdf > 0.0f & !occluded) {
                                auto wi = light_sample.shadow_ray->direction();
                                auto eval = closure->evaluate(wo, wi);
                                auto mis_weight = balance_heuristic(light_sample.eval.pdf, eval.pdf);
                                auto weight = mis_weight / light_sample.eval.pdf * beta;
                                Li -= weight * eval.f * light_sample.eval.L;

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                                $if(all(pixel_id == pixel_checked)) {
                                    auto Li_variation = weight * eval.f * light_sample.eval.L;
                                    pipeline().printer().info("direct lighting Li_variation = ({}, {}, {})",
                                                              Li_variation[0u], Li_variation[1u], Li_variation[2u]);
                                    pipeline().printer().info("after -direct: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                                };
#endif
                                closure->backward(wo, wi, d_loss * weight * light_sample.eval.L);
                                //auto surface_grad = closure->eval_grad(wo, wi);
                                //grad_map += surface_grad * weight * light_sample.eval.L;
                            };

                            // sample material
                            auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                            ray = it->spawn_ray(surface_sample.wi);
                            pdf_bsdf = surface_sample.eval.pdf;
                            auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);

                            // path replay bp
                            auto df = d_loss * grad_weight * Li;
                            df = ite(surface_sample.eval.f == 0.f, 0.f, df / surface_sample.eval.f);
                            closure->backward(wo, surface_sample.wi, df);

                            //auto surface_grad = closure->eval_grad(wo, surface_sample.wi);
                            //grad_map += ite(surface_sample.eval.f == 0.f, 0.f, surface_grad * Li / surface_sample.eval.f);

                            beta *= w * surface_sample.eval.f;

                            // apply eta scale
                            auto eta = closure->eta().value_or(1.f);
                            $switch(surface_sample.event) {
                                $case(Surface::event_enter) { eta_scale = sqr(eta); };
                                $case(Surface::event_exit) { eta_scale = 1.f / sqr(eta); };
                            };
                        };
                    });
                };
                // rr
                beta = zero_if_any_nan(beta);
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto rr_threshold = node<MegakernelReplayDiff>()->rr_threshold();
                auto q = max(beta.max() * eta_scale, .05f);
                $if(depth + 1u >= rr_depth) {
                    $if(q < rr_threshold & u_rr >= q) { $break; };
                    beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
                };
            };
            camera->film_grad()->accumulate(pixel_id, spectrum->srgb(swl, grad_map * shutter_weight));

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            $if(all(pixel_id == pixel_checked)) {
                pipeline().printer().info("should be 0: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
            };
#endif
        };
        auto bp_shader = pipeline().device().compile(bp_kernel);
        bp_shader_iter = _bp_shaders.emplace(camera, std::move(bp_shader)).first;
    }
    auto &&bp_shader = bp_shader_iter->second;
    command_buffer << synchronize();

    Clock clock;
    ProgressBar progress;
    progress.update(0.);
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 8u;
    auto sample_id = 0u;

    // de-correlate seed from the rendering part
    // Azinović, Tzu-Mao Li et al. [2019]
    // Inverse Path Tracing for Joint Material and Lighting Estimation
    
    auto seed_start = node<MegakernelReplayDiff>()->iterations() * spp;

    auto &&Li_1spp = replay_Li[camera];
    auto &&grad_in_real = grad_in;
    auto shutter_samples = camera->node()->shutter_samples();
    for (auto s : shutter_samples) {
        if (pipeline().update(command_buffer, s.point.time)) { dispatch_count = 0u; }
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render_1spp_shader(seed_start + iteration * spp + sample_id,
                                                 s.point.time, s.point.weight, Li_1spp)
                                  .dispatch(resolution) << synchronize();
            LUISA_INFO("forward propagation finished.");
            command_buffer << bp_shader(seed_start + iteration * spp + sample_id,
                                        s.point.time, s.point.weight, Li_1spp, grad_in_real)
                                  .dispatch(resolution)  << synchronize() ;
            sample_id++;
            dispatch_count += 2u;

            LUISA_INFO("backward propagation finished.");
            if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                // command_buffer << commit();
                dispatch_count -= dispatches_per_commit;
                auto p = sample_id / static_cast<double>(spp);
                command_buffer << [&progress, p] { progress.update(p); };
            }

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            command_buffer << pipeline().printer().retrieve() << synchronize();
#endif

        }
    }
    command_buffer << synchronize();
    progress.done();
    auto bp_time = clock.toc();
    LUISA_INFO("Backward propagation finished in {} ms", bp_time);
    LUISA_INFO("Start to accumulate gradients.");
    pipeline().differentiation()->accum_gradients(command_buffer);
    command_buffer << commit() << synchronize();
    LUISA_INFO("Step finished in {} ms", clock.toc()-bp_time);
}


// void MegakernelReplayDiffInstance::_render_one_camera_backward(
//     CommandBuffer &command_buffer, Camera::Instance *camera, BufferView<float4> grad_in) noexcept 
// {

//     //
//     //back propagate
//     /**/
// };

// void MegakernelReplayDiffInstance::_render_one_camera(
//     CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera) noexcept {

//     auto spp = camera->node()->spp();
//     auto resolution = camera->film()->node()->resolution();
//     auto image_file = camera->node()->file();

//     camera->film()->prepare(command_buffer);
//     if (!pipeline().has_lighting()) [[unlikely]] {
//         LUISA_WARNING_WITH_LOCATION(
//             "No lights in scene. Rendering aborted.");
//         return;
//     }

//     auto pt = this;
//     auto light_sampler = pt->light_sampler();
//     auto sampler = pt->sampler();
//     auto pixel_count = resolution.x * resolution.y;
//     sampler->reset(command_buffer, resolution, pixel_count, spp);
//     command_buffer << commit() << synchronize();

//     LUISA_INFO(
//         "Start rendering of resolution {}x{} at {}spp.",
//         resolution.x, resolution.y, spp);

//     auto shader_iter = _render_shaders.find(camera);
//     if (shader_iter == _render_shaders.end()) {
//         using namespace luisa::compute;

//         Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
//             set_block_size(16u, 16u, 1u);

//             auto pixel_id = dispatch_id().xy();
//             sampler->start(pixel_id, frame_index);
//             auto u_filter = sampler->generate_pixel_2d();
//             auto u_lens = camera->node()->requires_lens_sampling() ? sampler->generate_2d() : make_float2(.5f);
//             auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
//             auto spectrum = pipeline().spectrum();
//             auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
//             SampledSpectrum beta{swl.dimension(), camera_weight};
//             SampledSpectrum Li{swl.dimension()};

//             auto ray = camera_ray;
//             auto pdf_bsdf = def(1e16f);
//             $for(depth, pt->node<MegakernelReplayDiff>()->max_depth()) {

//                 // trace
//                 auto wo = -ray->direction();
//                 auto it = pipeline().geometry()->intersect(ray);

//                 // miss
//                 $if(!it->valid()) {
//                     if (pipeline().environment()) {
//                         auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
//                         Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
//                     }
//                     $break;
//                 };

//                 // hit light
//                 if (!pipeline().lights().empty()) {
//                     $if(it->shape().has_light()) {
//                         auto eval = light_sampler->evaluate_hit(
//                             *it, ray->origin(), swl, time);
//                         Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
//                     };
//                 }

//                 $if(!it->shape().has_surface()) { $break; };

//                 // sample one light
//                 auto u_light_selection = sampler->generate_1d();
//                 auto u_light_surface = sampler->generate_2d();
//                 auto u_lobe = sampler->generate_1d();
//                 auto u_bsdf = sampler->generate_2d();

//                 auto u_rr = def(0.f);
//                 auto rr_depth = node<MegakernelReplayDiff>()->rr_depth();
//                 $if(depth + 1u >= rr_depth) { u_rr = sampler->generate_1d(); };

//                 auto light_sample = LightSampler::Sample::zero(swl.dimension());
//                 $outline {
//                     // sample one light
//                     light_sample = light_sampler->sample(
//                         *it, u_light_selection, u_light_surface, swl, time);
//                 };

//                 // trace shadow ray
//                 auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

//                 // evaluate material
//                 auto surface_tag = it->shape().surface_tag();
//                 auto eta_scale = def(1.f);
//                 $outline {
//                     PolymorphicCall<Surface::Closure> call;
//                     pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
//                         surface->closure(call, *it, swl, wo, 1.f, time);
//                     });
//                     call.execute([&](const Surface::Closure *closure) noexcept {
//                         // apply opacity map
//                         auto alpha_skip = def(false);
//                         if (auto o = closure->opacity()) {
//                             auto opacity = saturate(*o);
//                             alpha_skip = u_lobe >= opacity;
//                             u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
//                         }

//                         $if(alpha_skip) {
//                             ray = it->spawn_ray(ray->direction());
//                             pdf_bsdf = 1e16f;
//                         }
//                         $else {
//                             if (auto dispersive = closure->is_dispersive()) {
//                                 $if(*dispersive) { swl.terminate_secondary(); };
//                             }

//                             // direct lighting
//                             $if(light_sample.eval.pdf > 0.0f & !occluded) {
//                                 auto wi = light_sample.shadow_ray->direction();
//                                 auto eval = closure->evaluate(wo, wi);
//                                 auto mis_weight = balance_heuristic(light_sample.eval.pdf, eval.pdf) / light_sample.eval.pdf;
//                                 Li += mis_weight * beta * eval.f * light_sample.eval.L;
//                             };
//                             // sample material
//                             auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
//                             ray = it->spawn_ray(surface_sample.wi);
//                             pdf_bsdf = surface_sample.eval.pdf;
//                             auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
//                             beta *= w * surface_sample.eval.f;

//                             // apply eta scale
//                             auto eta = closure->eta().value_or(1.f);
//                             $switch(surface_sample.event) {
//                                 $case(Surface::event_enter) { eta_scale = sqr(eta); };
//                                 $case(Surface::event_exit) { eta_scale = 1.f / sqr(eta); };
//                             };
//                         };
//                     });
//                 };

//                 // rr
//                 beta = zero_if_any_nan(beta);
//                 $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
//                 auto rr_threshold = node<MegakernelReplayDiff>()->rr_threshold();
//                 auto q = max(beta.max() * eta_scale, .05f);
//                 $if(depth + 1u >= rr_depth) {
//                     $if(q < rr_threshold & u_rr >= q) { $break; };
//                     beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
//                 };
//             };
//             camera->film()->accumulate(pixel_id, spectrum->srgb(swl, Li * shutter_weight));
//         };
//         auto render_shader = pipeline().device().compile(render_kernel);
//         shader_iter = _render_shaders.emplace(camera, std::move(render_shader)).first;
//     }
//     auto &&render_shader = shader_iter->second;
//     auto shutter_samples = camera->node()->shutter_samples();
//     command_buffer << synchronize();

//     Clock clock;
//     ProgressBar progress;
//     progress.update(0.);
//     auto dispatch_count = 0u;
//     auto dispatches_per_commit = 16u;
//     auto sample_id = 0u;
//     for (auto s : shutter_samples) {
//         if (pipeline().update(command_buffer, s.point.time)) {
//             dispatch_count = 0u;
//         }
//         for (auto i = 0u; i < s.spp; i++) {
//             command_buffer << render_shader(iteration * spp + sample_id++,
//                                             s.point.time, s.point.weight)
//                                   .dispatch(resolution);
//             camera->film()->show(command_buffer);
//             if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
//                 dispatch_count = 0u;
//                 auto p = sample_id / static_cast<double>(spp);
//                 command_buffer << [&progress, p] { progress.update(p); };
//             }
//         }
//     }
//     command_buffer << synchronize();
//     progress.done();

//     auto render_time = clock.toc();
//     LUISA_INFO("Rendering finished in {} ms.", render_time);
// }



// void render(Stream &stream) noexcept override {
//     auto pt = node<MegakernelReplayDiff>();
//     CommandBuffer command_buffer{&stream};
// #ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
//     command_buffer << pipeline().printer().reset();
// #endif
//     luisa::vector<float4> rendered;
//     luisa::vector<float4> rendered_last; // for computing finite-difference
//     luisa::vector<float4> rendered_delta;// for computing finite-difference
//     luisa::vector<float4> rendered_grad; // for grad_map

//     auto iteration_num = pt->iterations();

//     for (auto i = 0u; i < pipeline().camera_count(); i++) {
//         auto camera = pipeline().camera(i);

//         // delete output buffer
//         auto output_dir = std::filesystem::path("outputs") /
//                             luisa::format("output_buffer_camera_{:03}", i);
//         auto output_dir_2 = std::filesystem::path("outputs") /
//                             luisa::format("output_finite_diff_camera_{:03}", i);
//         auto output_dir_3 = std::filesystem::path("outputs") /
//                             luisa::format("output_grad_map_camera_{:03}", i);
//         // std::cout << "Current Path: " << output_dir << std::endl;
//         std::filesystem::remove_all(output_dir);
//         std::filesystem::remove_all(output_dir_2);
//         std::filesystem::remove_all(output_dir_3);
//         std::filesystem::create_directories(output_dir);
//         std::filesystem::create_directories(output_dir_2);
//         std::filesystem::create_directories(output_dir_3);
//     }

//     for (auto k = 0u; k < iteration_num; ++k) {
//         LUISA_INFO("");
//         LUISA_INFO("Iteration = {}", k);

//         // render
//         for (auto i = 0u; i < pipeline().camera_count(); i++) {
//             auto camera = pipeline().camera(i);
//             auto resolution = camera->film()->node()->resolution();
//             auto pixel_count = resolution.x * resolution.y;
//             auto output_path = std::filesystem::path("outputs") /
//                                 luisa::format("output_buffer_camera_{:03}", i) /
//                                 luisa::format("{:06}.exr", k);
//             auto output_path_gradmap = std::filesystem::path("outputs") /
//                                         luisa::format("output_grad_map_camera_{:03}", i) /
//                                         luisa::format("{:06}.exr", k);
//             auto output_path_finite_diff = std::filesystem::path("outputs") /
//                                             luisa::format("output_finite_diff_camera_{:03}", i) /
//                                             luisa::format("{:06}.exr", k);

//             LUISA_INFO("");
//             LUISA_INFO("Camera {}", i);
//             // render
//             _render_one_camera(command_buffer, k, camera);
//             // calculate grad
//             //_integrate_one_camera(command_buffer, k, camera);
//             // if (pt->save_process()) {
//             //     // save image
//             //     rendered.resize(next_pow2(pixel_count));
//             //     rendered_last.resize(next_pow2(pixel_count));
//             //     rendered_delta.resize(next_pow2(pixel_count));
//             //     rendered_grad.resize(next_pow2(pixel_count));
//             //     camera->film()->download(command_buffer, rendered.data());
//             //     command_buffer << synchronize();
//             //     save_image(output_path, (const float *)rendered.data(), resolution);
//             // }
            
//             // if (pt->save_grad_map()) {
//             //     camera->film_grad()->download(command_buffer, rendered_grad.data());

//             //     float max = -1000;
//             //     float min = 1000;
//             //     for (int i = 0; i < pixel_count; i++) {
//             //         if (rendered_grad[i].x > max)
//             //             max = rendered_grad[i].x;
//             //         if (rendered_grad[i].x < min)
//             //             min = rendered_grad[i].x;
//             //     }
//             //     auto max_abs = max > 0.f ? max : -max;
//             //     auto min_abs = min < 0.f ? -min : min;
//             //     auto bound = max_abs > min_abs ? max_abs : min_abs;
//             //     // LUISA_INFO("bound: {}", bound);
//             //     float satuation_up = 0.75;
//             //     float satuation_down = 0.1;

//             //     float saturatin_delta = satuation_up - satuation_down;
//             //     for (int i = 0; i < pixel_count; i++) {
//             //         rendered_grad[i].x = rendered_grad[i].x / bound * 3 * saturatin_delta;// scaling
//             //         if (rendered_grad[i].x < -saturatin_delta) {
//             //             // blue
//             //             rendered_grad[i].y = rendered_grad[i].x + satuation_up + saturatin_delta;
//             //             if (rendered_grad[i].y < satuation_down)
//             //                 rendered_grad[i].y = satuation_down;
//             //             rendered_grad[i].x = satuation_down;
//             //             rendered_grad[i].z = satuation_up;
//             //         } else if (rendered_grad[i].x < 0) {
//             //             // green
//             //             rendered_grad[i].z = satuation_down - rendered_grad[i].x;
//             //             rendered_grad[i].x = satuation_down;
//             //             rendered_grad[i].y = satuation_up;
//             //         } else if (rendered_grad[i].x < saturatin_delta) {
//             //             // yellow
//             //             rendered_grad[i].x = satuation_down + rendered_grad[i].x;
//             //             rendered_grad[i].y = satuation_up;
//             //             rendered_grad[i].z = satuation_down;
//             //         } else {
//             //             // red
//             //             rendered_grad[i].y = satuation_up + saturatin_delta - rendered_grad[i].x;
//             //             if (rendered_grad[i].y < satuation_down)
//             //                 rendered_grad[i].y = satuation_down;
//             //             rendered_grad[i].x = satuation_up;
//             //             rendered_grad[i].z = satuation_down;
//             //         }
//             //         rendered_grad[i].w = 1;
//             //     }
//             //     save_image(output_path_gradmap, (const float *)rendered_grad.data(), resolution);
//             //     }
//             //     if (pt->save_finite_diff()) {
//             //     // just toy case
//             //     auto &&last_time = last_time_rendered[camera];
//             //     if (k != 0) {
//             //         command_buffer << last_time.copy_to(rendered_last.data());
//             //         command_buffer << synchronize();
//             //         // float max = -1500;
//             //         // float min = 1500;
//             //         for (int i = 0; i < pixel_count; i++) {
//             //             auto pixel = rendered[i] - rendered_last[i];
//             //             rendered_delta[i].x = 0;// use idx x as a temp buffer
//             //             for (int j = 0; j < 3; j++) {
//             //                 rendered_delta[i].x += (rendered[i][j] - rendered_last[i][j]);
//             //             }
//             //             // if (i > 1024 * 130) {
//             //             //     if (rendered_delta[i].x > max)
//             //             //         max = rendered_delta[i].x;
//             //             //     if (rendered_delta[i].x < min)
//             //             //         min = rendered_delta[i].x;
//             //             // }
//             //         }
//             //         // LUISA_INFO("max: {}, min: {}", max, min);
//             //         for (int i = 0; i < pixel_count; i++) {
//             //             // if (rendered_delta[i].x > 20) {
//             //             //     LUISA_INFO("maxindex: {},{}", int(i / resolution.x), i - int(i / resolution.x) * resolution.x);
//             //             // }
//             //             rendered_delta[i].x = rendered_delta[i].x * 4;// scaling
//             //             if (rendered_delta[i].x < -1) {
//             //                 // blue
//             //                 rendered_delta[i].y = rendered_delta[i].x + 2;
//             //                 rendered_delta[i].x = 0;
//             //                 rendered_delta[i].z = 1;
//             //             } else if (rendered_delta[i].x < 0) {
//             //                 // green
//             //                 rendered_delta[i].z = 0 - rendered_delta[i].x;
//             //                 rendered_delta[i].x = 0;
//             //                 rendered_delta[i].y = 1;
//             //             } else if (rendered_delta[i].x < 1) {
//             //                 // yellow
//             //                 rendered_delta[i].y = 1;
//             //                 rendered_delta[i].z = 0;
//             //             } else {
//             //                 // red
//             //                 rendered_delta[i].y = 2 - rendered_delta[i].x;
//             //                 rendered_delta[i].x = 1;
//             //                 rendered_delta[i].z = 0;
//             //             }
//             //             rendered_delta[i].w = 1;
//             //         }
//             //         save_image(output_path_finite_diff, (const float *)rendered_delta.data(), resolution);
//             //     }
//             //     command_buffer << last_time.copy_from(rendered.data()) << synchronize();
//             //     }
//         }
//     }
//     // save results
//     LUISA_INFO("");
//     LUISA_INFO("Start to save results");
//     for (auto i = 0u; i < pipeline().camera_count(); i++) {
//         auto camera = pipeline().camera(i);
//         auto resolution = camera->film()->node()->resolution();
//         auto pixel_count = resolution.x * resolution.y;

//         _render_one_camera(command_buffer, iteration_num, camera);
//         // _render_grad_map(command_buffer, iteration_num, camera);

//         rendered.resize(next_pow2(pixel_count));
//         camera->film()->download(command_buffer, rendered.data());
//         command_buffer << compute::synchronize();
//         auto film_path = camera->node()->file();

//         save_image(film_path, (const float *)rendered.data(), resolution);
//     }
// #ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
//     command_buffer << pipeline().printer().retrieve() << synchronize();
// #endif
//     LUISA_INFO("Finish saving results");

//     // dump results of textured parameters
//     LUISA_INFO("");
//     LUISA_INFO("Dumping differentiable parameters");
//     pipeline().differentiation()->dump(command_buffer, "outputs");
//     LUISA_INFO("Finish dumping differentiable parameters");
// }

// #undef LUISA_RENDER_PATH_REPLAY_DEBUG

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelReplayDiff)

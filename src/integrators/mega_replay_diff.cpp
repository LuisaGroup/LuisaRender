//
// Created by ChenXin on 2022/2/23.
//

#include <util/imageio.h>

#include <luisa-compute.h>

#include <util/medium_tracker.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <core/stl.h>

namespace luisa::render {

#define LUISA_RENDER_PATH_REPLAY_DEBUG

using namespace luisa::compute;

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

class MegakernelReplayDiffInstance final : public DifferentiableIntegrator::Instance {

private:
    luisa::vector<float4> _pixels;
    luisa::optional<Window> _window;
    luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float>>
        _render_shaders;
    luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float, Image<float>>> _bp_shaders, _render_1spp_shaders;
    luisa::unordered_map<const Camera::Instance *, Image<float>> _Li;

private:
    void _render_one_camera(
        CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera,
        bool display = false) noexcept;

    void _integrate_one_camera(
        CommandBuffer &command_buffer, uint iteration, const Camera::Instance *camera) noexcept;

public:
    explicit MegakernelReplayDiffInstance(
        const MegakernelReplayDiff *node,
        Pipeline &pipeline, CommandBuffer &command_buffer) noexcept
        : DifferentiableIntegrator::Instance{pipeline, command_buffer, node} {

        // display
        if (node->display_camera_index() >= 0) {
            LUISA_ASSERT(node->display_camera_index() < pipeline.camera_count(),
                         "display_camera_index exceeds camera count");

            auto film = pipeline.camera(node->display_camera_index())->film()->node();
            _window.emplace("Display", film->resolution(), true);
            auto pixel_count = film->resolution().x * film->resolution().y;
            _pixels.resize(next_pow2(pixel_count) * 4u);
        }

        // save Li of the 2nd pass
        for (auto i = 0; i < pipeline.camera_count(); ++i) {
            auto camera = pipeline.camera(i);
            auto resolution = camera->film()->node()->resolution();
            _Li.emplace(camera, pipeline.device().create_image<float>(PixelStorage::FLOAT4, resolution));
        }

        // handle output dir
        std::filesystem::path output_dir{"outputs"};
        std::filesystem::remove_all(output_dir);
        std::filesystem::create_directories(output_dir);

        command_buffer << synchronize();
    }

    void display(CommandBuffer &command_buffer, const Film::Instance *film, uint iteration) noexcept {
        static auto exposure = 0.f;
        static auto aces = false;
        static auto a = 2.51f;
        static auto b = 0.03f;
        static auto c = 2.43f;
        static auto d = 0.59f;
        static auto e = 0.14f;
        if (_window) {
            if (_window->should_close()) {
                _window.reset();
                return;
            }
            _window->run_one_frame([&] {
                auto resolution = film->node()->resolution();
                auto pixel_count = resolution.x * resolution.y;
                film->download(command_buffer, _pixels.data());
                command_buffer << synchronize();
                auto scale = std::pow(2.f, exposure);
                auto pow = [](auto v, auto a) noexcept {
                    return make_float3(
                        std::pow(v.x, a),
                        std::pow(v.y, a),
                        std::pow(v.z, a));
                };
                auto tonemap = [](auto x) noexcept {
                    return x * (a * x + b) / (x * (c * x + d) + e);
                };
                for (auto &p : luisa::span{_pixels}.subspan(0u, pixel_count)) {
                    auto linear = scale * p.xyz();
                    if (aces) { linear = tonemap(linear); }
                    auto srgb = select(
                        1.055f * pow(linear, 1.0f / 2.4f) - 0.055f,
                        12.92f * linear,
                        linear <= 0.00304f);
                    p = make_float4(srgb, 1.f);
                }
                _window->set_background(_pixels.data(), resolution);
            });
        }
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<MegakernelReplayDiff>();
        auto command_buffer = stream.command_buffer();
#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
        command_buffer << pipeline().printer().reset();
#endif

        luisa::vector<float4> rendered;

        auto learning_rate = pt->learning_rate();
        auto iteration_num = pt->iterations();

        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);

            // delete output buffer
            auto output_dir = std::filesystem::path("outputs") /
                              luisa::format("output_buffer_camera_{:03}", i);
            std::filesystem::remove_all(output_dir);
            std::filesystem::create_directories(output_dir);
        }

        for (auto k = 0u; k < iteration_num; ++k) {
            auto loss = 0.f;

            LUISA_INFO("");
            LUISA_INFO("Iteration = {}", k);

            // render
            for (auto i = 0u; i < pipeline().camera_count(); i++) {
                auto camera = pipeline().camera(i);
                auto resolution = camera->film()->node()->resolution();
                auto pixel_count = resolution.x * resolution.y;
                auto output_path = std::filesystem::path("outputs") /
                                   luisa::format("output_buffer_camera_{:03}", i) /
                                   luisa::format("{:06}.exr", k);

                LUISA_INFO("");
                LUISA_INFO("Camera {}", i);

                // render
                _render_one_camera(command_buffer, k, camera, pt->display_camera_index() == i);

                // calculate grad
                _integrate_one_camera(command_buffer, k, camera);

                if (pt->save_process()) {
                    // save image
                    rendered.resize(next_pow2(pixel_count));
                    camera->film()->download(command_buffer, rendered.data());
                    command_buffer << synchronize();
                    save_image(output_path, (const float *)rendered.data(), resolution);
                }
            }

            // back propagate
            Clock clock;
            LUISA_INFO("");
            LUISA_INFO("Start to step");
            pipeline().differentiation().step(command_buffer, learning_rate);
            LUISA_INFO("Step finished in {} ms", clock.toc());
        }

        // save results
        LUISA_INFO("");
        LUISA_INFO("Start to save results");
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;

            _render_one_camera(command_buffer, iteration_num, camera);

            rendered.resize(next_pow2(pixel_count));
            camera->film()->download(command_buffer, rendered.data());
            command_buffer << compute::synchronize();
            auto film_path = camera->node()->file();

            save_image(film_path, (const float *)rendered.data(), resolution);
        }
#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
        command_buffer << pipeline().printer().retrieve() << synchronize();
#endif
        LUISA_INFO("Finish saving results");

        // dump results of textured parameters
        LUISA_INFO("");
        LUISA_INFO("Dumping differentiable parameters");
        pipeline().differentiation().dump(command_buffer, "outputs");
        LUISA_INFO("Finish dumping differentiable parameters");

        while (_window && !_window->should_close()) {
            _window->run_one_frame([] {});
        }
    }
};

luisa::unique_ptr<Integrator::Instance> MegakernelReplayDiff::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelReplayDiffInstance>(this, pipeline, command_buffer);
}

void MegakernelReplayDiffInstance::_integrate_one_camera(
    CommandBuffer &command_buffer, uint iteration, const Camera::Instance *camera) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->node()->film()->resolution();

    LUISA_INFO("Start backward propagation.");

    auto pt = this;

    auto sampler = pt->sampler();
    auto env = pipeline().environment();

    auto pixel_count = resolution.x * resolution.y;
    auto light_sampler = pt->light_sampler();
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();
    auto pt_exact = pt->node<MegakernelReplayDiff>();

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
    auto pixel_checked = make_uint2(120u, 280u);
#endif

    auto render_1spp_shader_iter = _render_1spp_shaders.find(camera);
    if (render_1spp_shader_iter == _render_1spp_shaders.end()) {
        using namespace luisa::compute;

        Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
            return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
        };

        Kernel2D render_kernel_1spp = [&](UInt frame_index, Float time, Float shutter_weight, ImageFloat Li_1spp) noexcept {
            set_block_size(16u, 16u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler->start(pixel_id, frame_index);
            auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel_id, time);
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
                auto it = pipeline().intersect(ray);

                // miss
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                        $if(all(pixel_id == pixel_checked)) {
                            pipeline().printer().info("Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                        };
#endif
                    }
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape()->has_light()) {
                        auto eval = light_sampler->evaluate_hit(
                            *it, ray->origin(), swl, time);
                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                        $if(all(pixel_id == pixel_checked)) {
                            pipeline().printer().info("Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                        };
#endif
                    };
                }

                $if(!it->shape()->has_surface()) { $break; };

                // sample one light
                Light::Sample light_sample = light_sampler->sample(
                    *sampler, *it, swl, time);

                // trace shadow ray
                auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
                auto occluded = pipeline().intersect_any(shadow_ray);

                // evaluate material
                SampledSpectrum eta_scale{swl.dimension(), 1.f};
                auto cos_theta_o = it->wo_local().z;
                auto surface_tag = it->shape()->surface_tag();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {
                    // apply roughness map
                    auto alpha_skip = def(false);
                    if (auto alpha_map = surface->alpha()) {
                        auto alpha = alpha_map->evaluate(*it, time).x;
                        alpha_skip = alpha < u_lobe;
                        u_lobe = ite(alpha_skip, (u_lobe - alpha) / (1.f - alpha), u_lobe / alpha);
                    }

                    $if(alpha_skip) {
                        ray = it->spawn_ray(ray->direction());
                        pdf_bsdf = 1e16f;
                    }
                    $else {
                        // create closure
                        auto closure = surface->closure(*it, swl, time);

                        // direct lighting
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.wi;
                            auto eval = closure->evaluate(wi);
                            auto mis_weight = balanced_heuristic(light_sample.eval.pdf, eval.pdf);
                            Li += mis_weight / light_sample.eval.pdf *
                                  abs_dot(eval.normal, wi) *
                                  beta * eval.f * light_sample.eval.L;

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                            $if(all(pixel_id == pixel_checked)) {
                                pipeline().printer().info("Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                            };
#endif
                        };

                        // sample material
                        auto sample = closure->sample(u_lobe, u_bsdf);
                        ray = it->spawn_ray(sample.wi);
                        pdf_bsdf = sample.eval.pdf;
                        auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);
                        beta *= abs(dot(sample.eval.normal, sample.wi)) * w * sample.eval.f;
                    };
                });

                // rr
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto q = max(spectrum->cie_y(swl, beta * eta_scale), .05f);
                auto rr_depth = pt->node<MegakernelReplayDiff>()->rr_depth();
                auto rr_threshold = pt->node<MegakernelReplayDiff>()->rr_threshold();
                $if(depth >= rr_depth & q < rr_threshold) {
                    $if(sampler->generate_1d() >= q) { $break; };
                    beta *= 1.0f / q;
                };
            };
            Li_1spp.write(pixel_id, make_float4(spectrum->srgb(swl, Li * shutter_weight), 1.f));

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            $if(all(pixel_id == pixel_checked)) {
                pipeline().printer().info("Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
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

        Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
            return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
        };

        Kernel2D bp_kernel = [&](UInt frame_index, Float time, Float shutter_weight, ImageFloat Li_1spp) noexcept {
            set_block_size(16u, 16u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler->start(pixel_id, frame_index);
            auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel_id, time);
            auto spectrum = pipeline().spectrum();
            auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension(), 0.f};
            auto grad_weight = shutter_weight * static_cast<float>(pt->node<MegakernelReplayDiff>()->max_depth());

            auto Li_last_pass = Li_1spp.read(pixel_id);
            Li[0u] = Li_last_pass[0u];
            Li[1u] = Li_last_pass[1u];
            Li[2u] = Li_last_pass[2u];

            SampledSpectrum d_loss{swl.dimension(), 0.f};
            auto d_loss_float3 = pt->loss()->d_loss(camera, pixel_id);
            for (auto i = 0u; i < 3u; ++i) {
                d_loss[i] = d_loss_float3[i];
            }

            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            $if(all(pixel_id == pixel_checked)) {
                pipeline().printer().info("Li_1spp backward: Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
            };
#endif

            $for(depth, pt->node<MegakernelReplayDiff>()->max_depth()) {

                // trace
                auto it = pipeline().intersect(ray);

                // miss, environment light
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler->evaluate_miss(
                            ray->direction(), swl, time);
                        Li -= beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                        $if(all(pixel_id == pixel_checked)) {
                            pipeline().printer().info("Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                        };
#endif
                    }
                    // TODO : backward environment light
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape()->has_light()) {
                        auto eval = light_sampler->evaluate_hit(
                            *it, ray->origin(), swl, time);
                        Li -= beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                        $if(all(pixel_id == pixel_checked)) {
                            pipeline().printer().info("Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                        };
#endif
                    };
                    // TODO : backward hit light
                }

                $if(!it->shape()->has_surface()) { $break; };

                // sample one light
                Light::Sample light_sample = light_sampler->sample(
                    *sampler, *it, swl, time);
                // trace shadow ray
                auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
                auto occluded = pipeline().intersect_any(shadow_ray);

                // evaluate material
                SampledSpectrum eta_scale{swl.dimension(), 1.f};
                auto cos_theta_o = it->wo_local().z;
                auto surface_tag = it->shape()->surface_tag();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {
                    // apply roughness map
                    auto alpha_skip = def(false);
                    if (auto alpha_map = surface->alpha()) {
                        auto alpha = alpha_map->evaluate(*it, time).x;
                        alpha_skip = alpha < u_lobe;
                        u_lobe = ite(alpha_skip, (u_lobe - alpha) / (1.f - alpha), u_lobe / alpha);
                    }

                    $if(alpha_skip) {
                        ray = it->spawn_ray(ray->direction());
                        pdf_bsdf = 1e16f;
                    }
                    $else {
                        // create closure
                        auto closure = surface->closure(*it, swl, time);

                        // direct lighting
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.wi;
                            auto eval = closure->evaluate(wi);
                            auto mis_weight = balanced_heuristic(light_sample.eval.pdf, eval.pdf);
                            auto weight = mis_weight / light_sample.eval.pdf * abs(dot(eval.normal, wi)) * beta;
                            Li -= weight * eval.f * light_sample.eval.L;

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
                            $if(all(pixel_id == pixel_checked)) {
                                auto Li_variation = weight * eval.f * light_sample.eval.L;
                                pipeline().printer().info("direct lighting Li_variation = ({}, {}, {})",
                                                         Li_variation[0u], Li_variation[1u], Li_variation[2u]);
                                pipeline().printer().info("Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
                            };
#endif

                            closure->backward(wi, d_loss * weight * light_sample.eval.L);

                            // TODO : backward direct light
                        };

                        // sample material
                        auto sample = closure->sample(u_lobe, u_bsdf);
                        ray = it->spawn_ray(sample.wi);
                        pdf_bsdf = sample.eval.pdf;
                        auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);

                        // path replay bp
                        auto df = d_loss * grad_weight * Li;
                        df = ite(sample.eval.f == 0.f, 0.f, df / sample.eval.f);
                        closure->backward(sample.wi, df);

                        beta *= abs(dot(sample.eval.normal, sample.wi)) * w * sample.eval.f;
                    };
                });

                // rr
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto q = max(spectrum->cie_y(swl, beta * eta_scale), .05f);
                auto rr_depth = pt->node<MegakernelReplayDiff>()->rr_depth();
                auto rr_threshold = pt->node<MegakernelReplayDiff>()->rr_threshold();
                $if(depth >= rr_depth & q < rr_threshold) {
                    $if(sampler->generate_1d() >= q) { $break; };
                    beta *= 1.0f / q;
                };
            };

#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            $if(all(pixel_id == pixel_checked)) {
                pipeline().printer().info("Li = ({}, {}, {})", Li[0u], Li[1u], Li[2u]);
            };
#endif
        };
        auto bp_shader = pipeline().device().compile(bp_kernel);
        bp_shader_iter = _bp_shaders.emplace(camera, std::move(bp_shader)).first;
    }
    auto &&bp_shader = bp_shader_iter->second;
    command_buffer << synchronize();

    Clock clock;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 8u;
    auto sample_id = 0u;

    // de-correlate seed from the rendering part
    // Azinović, Tzu-Mao Li et al. [2019]
    // Inverse Path Tracing for Joint Material and Lighting Estimation
    auto seed_start = node<MegakernelReplayDiff>()->iterations() * spp;

    auto &&Li_1spp = _Li[camera];
    auto shutter_samples = camera->node()->shutter_samples();
    for (auto s : shutter_samples) {
        if (pipeline().update(command_buffer, s.point.time)) { dispatch_count = 0u; }
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render_1spp_shader(seed_start + iteration * spp + sample_id,
                                                 s.point.time, s.point.weight, Li_1spp)
                                  .dispatch(resolution)
                           << bp_shader(seed_start + iteration * spp + sample_id++,
                                        s.point.time, s.point.weight, Li_1spp)
                                  .dispatch(resolution);
            dispatch_count += 2u;
            if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count -= dispatches_per_commit;
            }
#ifdef LUISA_RENDER_PATH_REPLAY_DEBUG
            command_buffer << pipeline().printer().retrieve() << synchronize();
#endif
        }
    }

    command_buffer << synchronize();
    LUISA_INFO("Backward propagation finished in {} ms.",
               clock.toc());
}

void MegakernelReplayDiffInstance::_render_one_camera(
    CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera,
    bool display) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    camera->film()->clear(command_buffer);
    if (!pipeline().has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto pt = this;
    auto light_sampler = pt->light_sampler();
    auto sampler = pt->sampler();
    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();

    LUISA_INFO(
        "Start rendering of resolution {}x{} at {}spp.",
        resolution.x, resolution.y, spp);

    auto shader_iter = _render_shaders.find(camera);
    if (shader_iter == _render_shaders.end()) {
        using namespace luisa::compute;

        Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
            return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
        };

        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler->start(pixel_id, frame_index);
            auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel_id, time);
            auto spectrum = pipeline().spectrum();
            auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension()};

            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);
            $for(depth, pt->node<MegakernelReplayDiff>()->max_depth()) {

                // trace
                auto it = pipeline().intersect(ray);

                // miss
                $if(!it->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                    }
                    $break;
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it->shape()->has_light()) {
                        auto eval = light_sampler->evaluate_hit(
                            *it, ray->origin(), swl, time);
                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                    };
                }

                $if(!it->shape()->has_surface()) { $break; };

                // sample one light
                Light::Sample light_sample = light_sampler->sample(
                    *sampler, *it, swl, time);

                // trace shadow ray
                auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
                auto occluded = pipeline().intersect_any(shadow_ray);

                // evaluate material
                SampledSpectrum eta_scale{swl.dimension(), 1.f};
                auto cos_theta_o = it->wo_local().z;
                auto surface_tag = it->shape()->surface_tag();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {
                    // apply roughness map
                    auto alpha_skip = def(false);
                    if (auto alpha_map = surface->alpha()) {
                        auto alpha = alpha_map->evaluate(*it, time).x;
                        alpha_skip = alpha < u_lobe;
                        u_lobe = ite(alpha_skip, (u_lobe - alpha) / (1.f - alpha), u_lobe / alpha);
                    }

                    $if(alpha_skip) {
                        ray = it->spawn_ray(ray->direction());
                        pdf_bsdf = 1e16f;
                    }
                    $else {
                        // create closure
                        auto closure = surface->closure(*it, swl, time);

                        // direct lighting
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.wi;
                            auto eval = closure->evaluate(wi);
                            auto mis_weight = balanced_heuristic(light_sample.eval.pdf, eval.pdf);
                            Li += mis_weight / light_sample.eval.pdf *
                                  abs_dot(eval.normal, wi) *
                                  beta * eval.f * light_sample.eval.L;
                        };

                        // sample material
                        auto sample = closure->sample(u_lobe, u_bsdf);
                        ray = it->spawn_ray(sample.wi);
                        pdf_bsdf = sample.eval.pdf;
                        auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);
                        beta *= abs(dot(sample.eval.normal, sample.wi)) * w * sample.eval.f;
                    };
                });

                // rr
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto q = max(spectrum->cie_y(swl, beta * eta_scale), .05f);
                auto rr_depth = pt->node<MegakernelReplayDiff>()->rr_depth();
                auto rr_threshold = pt->node<MegakernelReplayDiff>()->rr_threshold();
                $if(depth >= rr_depth & q < rr_threshold) {
                    $if(sampler->generate_1d() >= q) { $break; };
                    beta *= 1.0f / q;
                };
            };
            camera->film()->accumulate(pixel_id, spectrum->srgb(swl, Li * shutter_weight));
        };
        auto render_shader = pipeline().device().compile(render_kernel);
        shader_iter = _render_shaders.emplace(camera, std::move(render_shader)).first;
    }
    auto &&render_shader = shader_iter->second;
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    Clock clock;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 16u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        if (pipeline().update(command_buffer, s.point.time)) {
            dispatch_count = 0u;
        }
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render_shader(iteration * spp + sample_id++,
                                            s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count = 0u;
            }
        }
    }
    command_buffer << synchronize();
    LUISA_INFO("Rendering finished in {} ms.",
               clock.toc());
    if (display) { pt->display(command_buffer, camera->film(), iteration); }
}

#undef LUISA_RENDER_PATH_REPLAY_DEBUG

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelReplayDiff)

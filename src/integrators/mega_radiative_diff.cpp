//
// Created by ChenXin on 2022/2/23.
//

#include <luisa-compute.h>

#include <util/imageio.h>
#include <util/medium_tracker.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <core/stl.h>

namespace luisa::render {

using namespace luisa::compute;

class MegakernelRadiativeDiff final : public DifferentiableIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    MegakernelRadiativeDiff(Scene *scene, const SceneNodeDesc *desc) noexcept
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

class MegakernelRadiativeDiffInstance final : public DifferentiableIntegrator::Instance {

private:
    luisa::vector<float4> _pixels;
    luisa::optional<Window> _window;
    luisa::unordered_map<const Camera::Instance *, Shader<2, uint, float, float>>
        _bp_shaders, _render_shaders;

private:
    void _render_one_camera(
        CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera,
        bool display = false) noexcept;

    void _integrate_one_camera(
        CommandBuffer &command_buffer, uint iteration, const Camera::Instance *camera) noexcept;

public:
    explicit MegakernelRadiativeDiffInstance(
        const MegakernelRadiativeDiff *node,
        Pipeline &pipeline, CommandBuffer &command_buffer) noexcept
        : DifferentiableIntegrator::Instance{pipeline, command_buffer, node} {
        if (node->display_camera_index() >= 0) {
            LUISA_ASSERT(node->display_camera_index() < pipeline.camera_count(),
                         "display_camera_index exceeds camera count");

            auto film = pipeline.camera(node->display_camera_index())->film()->node();
            _window.emplace("Display", film->resolution(), true);
            auto pixel_count = film->resolution().x * film->resolution().y;
            _pixels.resize(next_pow2(pixel_count) * 4u);
        }

        std::filesystem::path output_dir{"outputs"};
        std::filesystem::remove_all(output_dir);
        std::filesystem::create_directories(output_dir);
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
        auto pt = node<MegakernelRadiativeDiff>();
        auto command_buffer = stream.command_buffer();
        luisa::vector<float4> rendered;

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
            LUISA_INFO("");
            LUISA_INFO("Start to step");
            Clock clock;
            pipeline().differentiation()->step(command_buffer);
            command_buffer << synchronize();
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
        LUISA_INFO("Finish saving results");

        // dump results of textured parameters
        LUISA_INFO("");
        LUISA_INFO("Dumping differentiable parameters");
        pipeline().differentiation()->dump(command_buffer, "outputs");
        LUISA_INFO("Finish dumping differentiable parameters");

        while (_window && !_window->should_close()) {
            _window->run_one_frame([] {});
        }
    }
};

luisa::unique_ptr<Integrator::Instance> MegakernelRadiativeDiff::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelRadiativeDiffInstance>(this, pipeline, command_buffer);
}

void MegakernelRadiativeDiffInstance::_integrate_one_camera(
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
    auto pt_exact = pt->node<MegakernelRadiativeDiff>();

    auto shader_iter = _bp_shaders.find(camera);
    if (shader_iter == _bp_shaders.end()) {
        using namespace luisa::compute;

        Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
            return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
        };

        Kernel2D bp_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);

            auto pixel_id = dispatch_id().xy();
            sampler->start(pixel_id, frame_index);
            auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel_id, time);
            auto spectrum = pipeline().spectrum();
            auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
//            SampledSpectrum beta{swl.dimension(), camera_weight * pixel_count};
            SampledSpectrum beta{swl.dimension(), camera_weight};
            SampledSpectrum Li{swl.dimension(), 1.0f};
            auto grad_weight = shutter_weight * static_cast<float>(pt->node<MegakernelRadiativeDiff>()->max_depth());

            auto d_loss = pt->loss()->d_loss(camera, pixel_id, swl);
            for (auto i = 0u; i < 3u; ++i) {
                beta[i] *= d_loss[i];
            }

            auto ray = camera_ray;
            auto pdf_bsdf = def(1e16f);

            $for(depth, pt->node<MegakernelRadiativeDiff>()->max_depth()) {

                // trace
                auto it = pipeline().geometry()->intersect(ray);

                // miss, environment light
                $if(!it->valid()) {
                    //                    if (pipeline.environment()) {
                    //                        auto eval = light_sampler->evaluate_miss(
                    //                            ray->direction(), env_to_world, swl, time);
                    //                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                    //                    }
                    //                    // TODO : backward environment light
                    $break;
                };

                //                // hit light
                //                if (!pipeline().lights().empty()) {
                //                    $if(it->shape()->has_light()) {
                //                        auto eval = light_sampler->evaluate_hit(
                //                            *it, ray->origin(), swl, time);
                //                        Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                //                    };
                //                    // TODO : backward hit light
                //                }

                $if(!it->shape()->has_surface()) { $break; };

                // sample one light
                auto u_light_selection = sampler->generate_1d();
                auto u_light_surface = sampler->generate_2d();
                Light::Sample light_sample = light_sampler->sample(
                    *it, u_light_selection, u_light_surface, swl, time);

                // trace shadow ray
                auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
                auto occluded = pipeline().geometry()->intersect_any(shadow_ray);

                // evaluate material
                auto surface_tag = it->shape()->surface_tag();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();
                auto eta_scale = def(1.f);
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {

                    // create closure
                    auto closure = surface->closure(*it, swl, 1.f, time);
                    if (auto dispersive = closure->is_dispersive()) {
                        $if (*dispersive) { swl.terminate_secondary(); };
                    }

                    // apply roughness map
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
                        auto wo = -ray->direction();

                        // direct lighting
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.wi;
                            auto eval = closure->evaluate(wo, wi);
                            auto mis_weight = balanced_heuristic(light_sample.eval.pdf, eval.pdf);
                            //                            Li += mis_weight / light_sample.eval.pdf *
                            //                                  abs_dot(eval.normal, wi) *
                            //                                  beta * eval.f * light_sample.eval.L;

                            // TODO : or apply the approximation light_sample.eval.L / light_sample.eval.pdf = 1.f
                            auto weight = mis_weight / light_sample.eval.pdf;
                            closure->backward(wo, wi, weight * beta * light_sample.eval.L);

                            // TODO : backward direct light
                        };

                        // sample material
                        auto sample = closure->sample(wo, u_lobe, u_bsdf);
                        ray = it->spawn_ray(sample.wi);
                        pdf_bsdf = sample.eval.pdf;
                        auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);

                        // radiative bp
                        // Li * d_fs
                        closure->backward(wo, sample.wi, grad_weight * beta * Li * w);

                        // d_Li * fs
                        beta *= w * sample.eval.f;

                        // apply eta scale
                        auto eta = closure->eta().value_or(1.f);
                        $switch (sample.event) {
                            $case (Surface::event_enter) { eta_scale = sqr(eta); };
                            $case (Surface::event_exit) { eta_scale = 1.f / sqr(eta); };
                        };
                    };
                });

                // rr
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto q = max(beta.max() * eta_scale, .05f);
                auto rr_depth = pt->node<MegakernelRadiativeDiff>()->rr_depth();
                auto rr_threshold = pt->node<MegakernelRadiativeDiff>()->rr_threshold();
                $if(depth + 1u >= rr_depth & q < rr_threshold) {
                    $if(sampler->generate_1d() >= q) { $break; };
                    beta *= 1.0f / q;
                };
            };
        };
        auto bp_shader = pipeline().device().compile(bp_kernel);
        shader_iter = _bp_shaders.emplace(camera, std::move(bp_shader)).first;
    }
    auto &&bp_shader = shader_iter->second;
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    Clock clock;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = 8u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        if (pipeline().update(command_buffer, s.point.time)) { dispatch_count = 0u; }
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << bp_shader(iteration * spp + sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count = 0u;
            }
        }
    }

    command_buffer << synchronize();
    LUISA_INFO("Backward propagation finished in {} ms",
               clock.toc());
}

void MegakernelRadiativeDiffInstance::_render_one_camera(
    CommandBuffer &command_buffer, uint iteration, Camera::Instance *camera,
    bool display) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    camera->film()->prepare(command_buffer);
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
            $for(depth, pt->node<MegakernelRadiativeDiff>()->max_depth()) {

                // trace
                auto it = pipeline().geometry()->intersect(ray);

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
                auto u_light_selection = sampler->generate_1d();
                auto u_light_surface = sampler->generate_2d();
                Light::Sample light_sample = light_sampler->sample(
                    *it, u_light_selection, u_light_surface, swl, time);

                // trace shadow ray
                auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
                auto occluded = pipeline().geometry()->intersect_any(shadow_ray);

                // evaluate material
                auto surface_tag = it->shape()->surface_tag();
                auto u_lobe = sampler->generate_1d();
                auto u_bsdf = sampler->generate_2d();
                auto eta_scale = def(1.f);
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {

                    // create closure
                    auto closure = surface->closure(*it, swl, 1.f, time);
                    if (auto dispersive = closure->is_dispersive()) {
                        $if (*dispersive) { swl.terminate_secondary(); };
                    }

                    // apply roughness map
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

                        auto wo = -ray->direction();

                        // direct lighting
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.wi;
                            auto eval = closure->evaluate(wo, wi);
                            auto mis_weight = balanced_heuristic(light_sample.eval.pdf, eval.pdf);
                            Li += mis_weight / light_sample.eval.pdf * beta * eval.f * light_sample.eval.L;
                        };

                        // sample material
                        auto sample = closure->sample(wo, u_lobe, u_bsdf);
                        ray = it->spawn_ray(sample.wi);
                        pdf_bsdf = sample.eval.pdf;
                        auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);
                        beta *= w * sample.eval.f;

                        // apply eta scale
                        auto eta = closure->eta().value_or(1.f);
                        $switch (sample.event) {
                            $case (Surface::event_enter) { eta_scale = sqr(eta); };
                            $case (Surface::event_exit) { eta_scale = 1.f / sqr(eta); };
                        };
                    };
                });

                // rr
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto q = max(beta.max() * eta_scale, .05f);
                auto rr_depth = pt->node<MegakernelRadiativeDiff>()->rr_depth();
                auto rr_threshold = pt->node<MegakernelRadiativeDiff>()->rr_threshold();
                $if(depth + 1u >= rr_depth & q < rr_threshold) {
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
    LUISA_INFO("Rendering finished in {} ms",
               clock.toc());
    if (display) { pt->display(command_buffer, camera->film(), iteration); }
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelRadiativeDiff)

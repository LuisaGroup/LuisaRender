//
// Created by Mike Smith on 2022/1/10.
//

#include <util/imageio.h>
#include <luisa-compute.h>

#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

using namespace compute;

class MegakernelPathTracing final : public Integrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    bool _display;

public:
    MegakernelPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _display{desc->property_bool_or_default("display")} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return false; }
    [[nodiscard]] bool display_enabled() const noexcept { return _display; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPathTracingInstance final : public Integrator::Instance {

private:
    uint _last_spp{0u};
    Clock _clock;
    Framerate _framerate;
    luisa::vector<float4> _pixels;
    luisa::optional<Window> _window;

private:
    static void _render_one_camera(
        CommandBuffer &command_buffer, Pipeline &pipeline,
        MegakernelPathTracingInstance *pt, Camera::Instance *camera) noexcept;

public:
    explicit MegakernelPathTracingInstance(const MegakernelPathTracing *node, Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {
        if (node->display_enabled()) {
            auto first_film = pipeline.camera(0u)->film()->node();
            _window.emplace("Display", first_film->resolution(), true);
        }
    }

    void display(CommandBuffer &command_buffer, const Film::Instance *film, uint spp) noexcept {
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
            _framerate.record(spp - _last_spp);
            _last_spp = spp;
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
                ImGui::Begin("Console", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::Text("Frame: %u", spp);
                ImGui::Text("Time: %.1fs", _clock.toc() * 1e-3);
                ImGui::Text("FPS: %.2f", _framerate.report());
                ImGui::Checkbox("ACES", &aces);
                if (aces) {
                    ImGui::SameLine();
                    ImGui::Spacing();
                    ImGui::SameLine();
                    if (ImGui::Button("Reset")) {
                        a = 2.51f;
                        b = 0.03f;
                        c = 2.43f;
                        d = 0.59f;
                        e = 0.14f;
                    }
                    ImGui::SliderFloat("A", &a, 0.f, 3.f, "%.2f");
                    ImGui::SliderFloat("B", &b, 0.f, 3.f, "%.2f");
                    ImGui::SliderFloat("C", &c, 0.f, 3.f, "%.2f");
                    ImGui::SliderFloat("D", &d, 0.f, 3.f, "%.2f");
                    ImGui::SliderFloat("E", &e, 0.f, 3.f, "%.2f");
                }
                ImGui::SliderFloat("Exposure", &exposure, -10.f, 10.f, "%.1f");
                ImGui::End();
                _window->set_background(_pixels.data(), resolution);
            });
        }
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<MegakernelPathTracing>();
        auto command_buffer = stream.command_buffer();
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _last_spp = 0u;
            _clock.tic();
            _framerate.clear();
            _pixels.resize(next_pow2(pixel_count) * 4u);
            _render_one_camera(command_buffer, pipeline(), this, camera);
            camera->film()->download(command_buffer, _pixels.data());
            command_buffer << compute::synchronize();
            auto film_path = camera->node()->file();
            save_image(film_path, (const float *)_pixels.data(), resolution);
        }
        while (_window && !_window->should_close()) {
            _window->run_one_frame([] {});
        }
    }
};

unique_ptr<Integrator::Instance> MegakernelPathTracing::build(Pipeline &pipeline, CommandBuffer &cmd_buffer) const noexcept {
    return luisa::make_unique<MegakernelPathTracingInstance>(this, pipeline, cmd_buffer);
}

void MegakernelPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Pipeline &pipeline,
    MegakernelPathTracingInstance *pt, Camera::Instance *camera) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    camera->film()->clear(command_buffer);
    if (!pipeline.has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto light_sampler = pt->light_sampler();
    auto sampler = pt->sampler();
    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();

    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
    };

    Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(8u, 8u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel_id, time);
        auto swl = pt->spectrum()->sample(*sampler);
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        $for(depth, pt->node<MegakernelPathTracing>()->max_depth()) {

            // trace
            auto it = pipeline.intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline.environment()) {
                    auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balanced_heuristic(pdf_bsdf, eval.pdf);
                }
                $break;
            };

            // hit light
            if (!pipeline.lights().empty()) {
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
            auto occluded = pipeline.intersect_any(shadow_ray);

            // evaluate material
            SampledSpectrum eta_scale{swl.dimension(), 1.f};
            auto surface_tag = it->shape()->surface_tag();
            auto u_lobe = sampler->generate_1d();
            auto u_bsdf = sampler->generate_2d();
            pipeline.dynamic_dispatch_surface(surface_tag, [&](auto surface) noexcept {
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
                        Li += mis_weight / light_sample.eval.pdf * abs(dot(eval.normal, wi)) *
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
            $if(beta.all([](auto b) noexcept { return isnan(b) | b <= 0.f; })) { $break; };
            auto q = max(swl.cie_y(beta), .05f);
            auto rr_depth = pt->node<MegakernelPathTracing>()->rr_depth();
            auto rr_threshold = pt->node<MegakernelPathTracing>()->rr_threshold();
            $if(depth >= rr_depth & q < rr_threshold) {
                $if(sampler->generate_1d() >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        camera->film()->accumulate(pixel_id, swl.srgb(Li * shutter_weight));
    };
    auto render = pipeline.device().compile(render_kernel);
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    LUISA_INFO("Rendering started.");
    ProgressBar progress;
    progress.update(0.0);

    auto display = pt->node<MegakernelPathTracing>()->display_enabled();
    auto dispatch_count = 0u;
    auto dispatches_per_commit = display ? 4u : 32u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        pipeline.update(command_buffer, s.point.time);
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                dispatch_count = 0u;
                auto p = sample_id / static_cast<double>(spp);
                if (display) {
                    pt->display(command_buffer, camera->film(), sample_id);
                    progress.update(p);
                } else {
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
        }
    }
    command_buffer << synchronize();
    progress.done();
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathTracing)

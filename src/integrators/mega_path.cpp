//
// Created by Mike Smith on 2022/1/10.
//

#include <tinyexr.h>
#include <luisa-compute.h>

#include <util/medium_tracker.h>
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
    [[nodiscard]] bool differentiable() const noexcept override { return false; }
    [[nodiscard]] bool display_enabled() const noexcept { return _display; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPathTracingInstance final : public Integrator::Instance {

private:
    Pipeline &_pipeline;
    uint _last_spp{0u};
    Clock _clock;
    Framerate _framerate;
    luisa::vector<float4> _pixels;
    luisa::optional<Window> _window;

private:
    static void _render_one_camera(
        CommandBuffer &command_buffer, Pipeline &pipeline, MegakernelPathTracingInstance *pt,
        const Camera::Instance *camera, const Filter::Instance *filter, Film::Instance *film) noexcept;

public:
    explicit MegakernelPathTracingInstance(const MegakernelPathTracing *node, Pipeline &pipeline) noexcept
        : Integrator::Instance{pipeline, node}, _pipeline{pipeline} {
        if (node->display_enabled()) {
            auto first_film = get<1>(_pipeline.camera(0u))->node();
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
        for (auto i = 0u; i < _pipeline.camera_count(); i++) {
            auto [camera, film, filter] = _pipeline.camera(i);
            auto resolution = film->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _last_spp = 0u;
            _clock.tic();
            _framerate.clear();
            _pixels.resize(next_pow2(pixel_count) * 4u);
            _render_one_camera(
                command_buffer, _pipeline,
                this, camera, filter, film);
            film->download(command_buffer, _pixels.data());
            command_buffer << compute::synchronize();
            auto film_path = camera->node()->file();
            if (film_path.extension() != ".exr") [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unexpected film file extension. "
                    "Changing to '.exr'.");
                film_path.replace_extension(".exr");
            }
            auto size = make_int2(resolution);
            const char *err = nullptr;
            SaveEXR(reinterpret_cast<const float *>(_pixels.data()),
                    size.x, size.y, 4, false, film_path.string().c_str(), &err);
            if (err != nullptr) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Failed to save film to '{}'.",
                    film_path.string());
            }
        }
        while (_window && !_window->should_close()) {
            _window->run_one_frame([] {});
        }
    }
};

unique_ptr<Integrator::Instance> MegakernelPathTracing::build(Pipeline &pipeline, CommandBuffer &) const noexcept {
    return luisa::make_unique<MegakernelPathTracingInstance>(this, pipeline);
}

void MegakernelPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Pipeline &pipeline, MegakernelPathTracingInstance *pt,
    const Camera::Instance *camera, const Filter::Instance *filter, Film::Instance *film) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = film->node()->resolution();
    auto image_file = camera->node()->file();
    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    auto light_sampler = pipeline.light_sampler();
    auto sampler = pipeline.sampler();
    auto env = pipeline.environment();

    film->clear(command_buffer);
    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer.commit();

    using namespace luisa::compute;
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        return ite(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), 0.0f);
    };

    Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 camera_to_world_normal,
                                 Float3x3 env_to_world, Float time, Float shutter_weight) noexcept {
        set_block_size(8u, 8u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto pixel = make_float2(pixel_id) + 0.5f;
        auto beta = def(make_float4(1.f));
        auto [filter_offset, filter_weight] = filter->sample(*sampler);
        pixel += filter_offset;
        beta *= filter_weight;
        auto swl = SampledWavelengths::sample_visible(sampler->generate_1d());
        auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel, time);
        if (!camera->node()->transform()->is_identity()) {
            camera_ray->set_origin(make_float3(camera_to_world * make_float4(camera_ray->origin(), 1.0f)));
            camera_ray->set_direction(normalize(camera_to_world_normal * camera_ray->direction()));
        }
        beta *= camera_weight;

        auto ray = camera_ray;
        auto Li = def(make_float4(0.0f));
        auto pdf_bsdf = def(0.0f);
        $for(depth, pt->node<MegakernelPathTracing>()->max_depth()) {

            auto add_light_contrib = [&](const Light::Evaluation &eval) noexcept {
                auto mis_weight = ite(depth == 0u, 1.0f, balanced_heuristic(pdf_bsdf, eval.pdf));
                Li += ite(eval.pdf > 0.0f, beta * eval.L * mis_weight, make_float4(0.0f));
            };

            auto env_prob = env == nullptr ? 0.0f : env->selection_prob();

            // trace
            auto it = pipeline.intersect(ray);

            // miss
            $if(!it->valid()) {
                if (env_prob > 0.0f) {
                    auto eval = env->evaluate(ray->direction(), env_to_world, swl, time);
                    eval.L /= env_prob;
                    add_light_contrib(eval);
                }
                $break;
            };

            // hit light
            if (light_sampler != nullptr && env_prob < 1.f) {
                $if(it->shape()->has_light()) {
                    auto eval = light_sampler->evaluate(*it, ray->origin(), swl, time);
                    eval.L /= 1.0f - env_prob;
                    add_light_contrib(eval);
                };
            }

            // sample one light
            $if(!it->shape()->has_surface()) { $break; };
            Light::Sample light_sample;
            if (env_prob > 0.0f) {
                auto u = sampler->generate_1d();
                $if(u < env_prob) {
                    light_sample = env->sample(*sampler, it->p(), env_to_world, swl, time);
                    light_sample.eval.pdf *= env_prob;
                }
                $else {
                    if (light_sampler != nullptr) {
                        light_sample = light_sampler->sample(*sampler, *it, swl, time);
                        light_sample.eval.pdf *= 1.0f - env_prob;
                    }
                };
            } else if (light_sampler != nullptr) {
                light_sample = light_sampler->sample(*sampler, *it, swl, time);
                light_sample.eval.pdf *= 1.0f - env_prob;
            }

            // trace shadow ray
            auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
            auto occluded = pipeline.intersect_any(shadow_ray);

            // evaluate material
            auto eta_scale = def(make_float4(1.f));
            auto cos_theta_o = it->wo_local().z;
            auto surface_tag = it->shape()->surface_tag();
            pipeline.dynamic_dispatch_surface(surface_tag, [&](auto surface) {
                // apply normal map
                if (auto normal_map = surface->normal()) {
                    auto normal_local = 2.f * normal_map->evaluate(*it, swl, time).value.xyz() - 1.f;
                    auto normal = it->shading().local_to_world(normal_local);
                    it->set_shading(Frame::make(normal, it->shading().u()));
                }
                // apply alpha map
                auto alpha_skip = def(false);
                if (auto alpha_map = surface->alpha()) {
                    auto alpha = alpha_map->evaluate(*it, swl, time).value.x;
                    auto u_alpha = sampler->generate_1d();
                    alpha_skip = alpha < u_alpha;
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
                        auto cos_theta_i = dot(it->shading().n(), wi);
                        auto is_trans = cos_theta_i * cos_theta_o < 0.f;
                        auto mis_weight = balanced_heuristic(light_sample.eval.pdf, eval.pdf);
                        Li += beta * mis_weight * ite(eval.pdf > 0.0f, eval.f, 0.0f) *
                              abs_dot(it->shading().n(), wi) *
                              light_sample.eval.L / light_sample.eval.pdf;
                    };

                    // sample material
                    auto [wi, eval] = closure->sample(*sampler);
                    auto cos_theta_i = dot(wi, it->shading().n());
                    ray = it->spawn_ray(wi);
                    pdf_bsdf = eval.pdf;
                    beta *= ite(
                        eval.pdf > 0.0f,
                        eval.f * abs(cos_theta_i) / eval.pdf,
                        make_float4(0.0f));
                    eta_scale = ite(
                        cos_theta_i * cos_theta_o < 0.f &
                            min(eval.alpha.x, eval.alpha.y) < .05f,
                        ite(cos_theta_o > 0.f, sqr(eval.eta), sqrt(1.f / eval.eta)),
                        1.f);
                };
            });

            // rr
            $if(all(beta <= 0.0f)) { $break; };
            auto q = max(swl.cie_y(beta * eta_scale), .05f);
            auto rr_depth = pt->node<MegakernelPathTracing>()->rr_depth();
            auto rr_threshold = pt->node<MegakernelPathTracing>()->rr_threshold();
            $if(depth >= rr_depth & q < rr_threshold) {
                $if(sampler->generate_1d() >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        film->accumulate(pixel_id, swl.srgb(Li * shutter_weight));
    };
    auto render = pipeline.device().compile(render_kernel);
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    auto display = pt->node<MegakernelPathTracing>()->display_enabled();

    Clock clock;
    auto dispatch_count = 0u;
    auto dispatches_per_commit = display ? 4u : 16u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        if (pipeline.update_geometry(command_buffer, s.point.time)) {
            dispatch_count = 0u;
            if (display) { pt->display(command_buffer, film, sample_id); }
        }
        auto camera_to_world = camera->node()->transform()->matrix(s.point.time);
        auto camera_to_world_normal = transpose(inverse(make_float3x3(camera_to_world)));
        auto env_to_world = env == nullptr || env->node()->transform()->is_identity() ?
                                make_float3x3(1.0f) :
                                transpose(inverse(make_float3x3(
                                    env->node()->transform()->matrix(s.point.time))));
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, camera_to_world, camera_to_world_normal,
                                     env_to_world, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                command_buffer << commit();
                dispatch_count = 0u;
                if (display) { pt->display(command_buffer, film, sample_id); }
            }
        }
    }
    command_buffer << synchronize();
    LUISA_INFO("Rendering finished in {} ms.", clock.toc());
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathTracing)

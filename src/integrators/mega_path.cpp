//
// Created by Mike Smith on 2022/1/10.
//

#include <fstream>
#include <util/imageio.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/display.h>

namespace luisa::render {

using namespace compute;

class MegakernelPathTracing final : public Integrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _display_interval;
    bool _display;

public:
    MegakernelPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _display{desc->property_bool_or_default("display")},
          _display_interval{desc->property_uint_or_default("display_interval", 1u)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto display_enabled() const noexcept { return _display; }
    [[nodiscard]] auto display_interval() const noexcept { return _display_interval; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPathTracingInstance final : public Integrator::Instance {

private:
    luisa::unique_ptr<Display> _display;

private:
    void _render_one_camera(
        CommandBuffer &command_buffer,
        Pipeline &pipeline,
        Camera::Instance *camera) noexcept;

public:
    explicit MegakernelPathTracingInstance(const MegakernelPathTracing *node,
                                           Pipeline &pipeline,
                                           CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {
        if (node->display_enabled()) {
            auto first_film = pipeline.camera(0u)->film()->node();
            _display = luisa::make_unique<Display>("Display");
        }
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<MegakernelPathTracing>();
        auto command_buffer = stream.command_buffer();
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            camera->film()->prepare(command_buffer);
            if (_display) { _display->reset(command_buffer, camera->film()); }
            _render_one_camera(command_buffer, pipeline(), camera);
            luisa::vector<float4> pixels(pixel_count);
            camera->film()->download(command_buffer, pixels.data());
            command_buffer << compute::synchronize();
            camera->film()->release();
            auto film_path = camera->node()->file();
            save_image(film_path, reinterpret_cast<const float *>(pixels.data()), resolution);
        }
        while (_display && _display->idle()) {}
    }
};

luisa::unique_ptr<Integrator::Instance> MegakernelPathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelPathTracingInstance>(
        this, pipeline, command_buffer);
}

void MegakernelPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Pipeline &pipeline, Camera::Instance *camera) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    if (!pipeline.has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto pixel_count = resolution.x * resolution.y;
    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << synchronize();

    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;

    Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler()->start(pixel_id, frame_index);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(*sampler(), pixel_id, time);
        auto spectrum = pipeline.spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        $for(depth, node<MegakernelPathTracing>()->max_depth()) {

            // trace
            auto it = pipeline.geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline.environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                }
                $break;
            };

            // hit light
            if (!pipeline.lights().empty()) {
                $if(it->shape()->has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                };
            }

            $if(!it->shape()->has_surface()) { $break; };

            // sample one light
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            Light::Sample light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline.geometry()->intersect_any(light_sample.ray);

            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto eta = def(1.f);
            auto eta_scale = def(1.f);
            auto alpha_skip = def(false);
            auto wo = -ray->direction();
            auto surface_sample = Surface::Sample::zero(swl.dimension());
            pipeline.surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(*it, swl, 1.f, time);

                // apply roughness map
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(!alpha_skip) {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                    };
                    // sample material
                    surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    eta = closure->eta().value_or(1.f);
                };
            });

            $if(alpha_skip) {
                ray = it->spawn_ray(ray->direction());
                pdf_bsdf = 1e16f;
            }
            $else {
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                beta *= w * surface_sample.eval.f;
                // apply eta scale
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta); };
                    $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                };
                // rr
                beta = zero_if_any_nan(beta);
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto rr_depth = node<MegakernelPathTracing>()->rr_depth();
                auto rr_threshold = node<MegakernelPathTracing>()->rr_threshold();
                auto q = max(beta.max() * eta_scale, .05f);
                $if(depth + 1u >= rr_depth) {
                    auto u = sampler()->generate_1d();
                    $if(q < rr_threshold & u >= q) { $break; };
                    beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
                };
            };
        };
        camera->film()->accumulate(pixel_id, spectrum->srgb(swl, Li * shutter_weight));
    };

    Clock clock_compile;
    auto render = pipeline.device().compile(render_kernel);
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    LUISA_INFO("Rendering started.");
    Clock clock;
    ProgressBar progress;
    progress.update(0.0);
    auto dispatch_count = 0u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        pipeline.update(command_buffer, s.point.time);
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            auto dispatches_per_commit =
                _display && !_display->should_close() ?
                    node<MegakernelPathTracing>()->display_interval() :
                    32u;
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                dispatch_count = 0u;
                auto p = sample_id / static_cast<double>(spp);
                if (_display && _display->update(command_buffer, sample_id)) {
                    progress.update(p);
                } else {
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
        }
    }
    command_buffer << synchronize();
    progress.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathTracing)

//
// Created by Mike Smith on 2022/1/10.
//

#include <fstream>
#include <luisa-compute.h>
#include <util/imageio.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

using namespace compute;

class DirectLighting final : public Integrator {

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
        : Integrator{scene, desc} {
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
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class DirectLightingInstance final : public Integrator::Instance {

private:
    luisa::vector<float4> _pixels;

private:
    static void _render_one_camera(
        CommandBuffer &command_buffer, Pipeline &pipeline,
        DirectLightingInstance *pt, Camera::Instance *camera) noexcept;

public:
    explicit DirectLightingInstance(const DirectLighting *node, Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {}

    void render(Stream &stream) noexcept override {
        auto pt = node<DirectLighting>();
        auto command_buffer = stream.command_buffer();
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _pixels.resize(next_pow2(pixel_count) * 4u);
            camera->film()->prepare(command_buffer);
            _render_one_camera(command_buffer, pipeline(), this, camera);
            camera->film()->download(command_buffer, _pixels.data());
            command_buffer << compute::synchronize();
            camera->film()->release();
            auto film_path = camera->node()->file();
            save_image(film_path, (const float *)_pixels.data(), resolution);
        }
    }
};

luisa::unique_ptr<Integrator::Instance> DirectLighting::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<DirectLightingInstance>(
        this, pipeline, command_buffer);
}

void DirectLightingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Pipeline &pipeline,
    DirectLightingInstance *pt, Camera::Instance *camera) noexcept {

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    if (!pipeline.has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto light_sampler = pt->light_sampler();
    auto sampler = pt->sampler();
    auto pixel_count = resolution.x * resolution.y;
    sampler->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << synchronize();

    // importance sampling strategy
    auto importance_sampling = pt->node<DirectLighting>()->importance_sampling();
    auto samples_lights = importance_sampling == DirectLighting::ImportanceSampling::LIGHT ||
                          importance_sampling == DirectLighting::ImportanceSampling::BOTH;
    auto samples_surfaces = importance_sampling == DirectLighting::ImportanceSampling::SURFACE ||
                            importance_sampling == DirectLighting::ImportanceSampling::BOTH;

    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;

    Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto cs = camera->generate_ray(*sampler, pixel_id, time);
        auto spectrum = pipeline.spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
        SampledSpectrum Li{swl.dimension(), 0.f};

        auto ray = cs.ray;

        $loop {

            // trace
            auto it = pipeline.geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline.environment()) {
                    auto eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                    Li += cs.weight * eval.L;
                }
                $break;
            };

            // hit light
            if (!pipeline.lights().empty()) {
                $if(it->shape()->has_light()) {
                    auto eval = light_sampler->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += cs.weight * eval.L;
                };
            }

            // compute direct lighting
            $if(!it->shape()->has_surface()) { $break; };

            auto light_sample = Light::Sample::zero(swl.dimension());
            auto occluded = def(false);

            if (samples_lights) {
                // sample one light
                auto u_light_selection = sampler->generate_1d();
                auto u_light_surface = sampler->generate_2d();
                light_sample = light_sampler->sample(
                    *it, u_light_selection, u_light_surface, swl, time);

                // trace shadow ray
                occluded = pipeline.geometry()->intersect_any(light_sample.ray);
            }

            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto u_lobe = sampler->generate_1d();
            auto u_bsdf = def(make_float2());
            if (samples_surfaces) { u_bsdf = sampler->generate_2d(); }
            auto surface_sample = Surface::Sample::zero(swl.dimension());
            auto alpha_skip = def(false);
            pipeline.surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(*it, swl, 1.f, time);

                // apply roughness map
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
                    auto wo = -ray->direction();
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    if (samples_lights) {
                        $if(light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.ray->direction();
                            auto eval = closure->evaluate(wo, wi);
                            $if (eval.pdf > 0.f) {
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

            $if (!alpha_skip) {
                if (samples_surfaces) {
                    // trace
                    auto bsdf_it = pipeline.geometry()->intersect(ray);

                    // miss
                    auto light_eval = Light::Evaluation::zero(swl.dimension());
                    $if(!bsdf_it->valid()) {
                        if (pipeline.environment()) {
                            light_eval = light_sampler->evaluate_miss(ray->direction(), swl, time);
                        }
                    } $else {
                        // hit light
                        if (!pipeline.lights().empty()) {
                            $if(bsdf_it->shape()->has_light()) {
                                light_eval = light_sampler->evaluate_hit(*bsdf_it, ray->origin(), swl, time);
                            };
                        }
                    };

                    $if (light_eval.pdf > 0.f & surface_sample.eval.pdf > 0.f) {
                        auto w = def(1.f);
                        if (samples_lights) { w = balance_heuristic(surface_sample.eval.pdf, light_eval.pdf); }
                        Li += cs.weight * w * surface_sample.eval.f * light_eval.L / surface_sample.eval.pdf;
                    };
                }
                $break;
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
    auto dispatches_per_commit = 32u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        pipeline.update(command_buffer, s.point.time);
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                dispatch_count = 0u;
                auto p = sample_id / static_cast<double>(spp);
                command_buffer << [&progress, p] { progress.update(p); };
            }
        }
    }
    command_buffer << synchronize();
    progress.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DirectLighting)

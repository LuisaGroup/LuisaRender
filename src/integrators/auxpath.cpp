//
// Created by Mike Smith on 2022/1/10.
//

#include <fstream>

#include <core/json.h>
#include <ast/function_serializer.h>
#include <util/imageio.h>
#include <luisa-compute.h>

#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/scene.h>

namespace luisa::render {

using namespace compute;

class AuxiliaryBufferPathTracing final : public Integrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _noisy_count;
    const Sampler *_aux_sampler;

public:
    AuxiliaryBufferPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _noisy_count{std::max(desc->property_uint_or_default("noisy_count", 4u), 4u)},
          _aux_sampler{scene->load_sampler(desc->property_node_or_default(
              "auxiliary_sampler", SceneNodeDesc::shared_default_sampler("independent")))} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto noisy_count() const noexcept { return _noisy_count; }
    [[nodiscard]] auto aux_sampler() const noexcept { return _aux_sampler; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return false; }
    [[nodiscard]] bool display_enabled() const noexcept { return false; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class AuxiliaryBufferPathTracingInstance final : public Integrator::Instance {

    const int AUXILIARY_BUFFER_COUNT = 18;

private:
    uint _last_spp{0u};
    Clock _clock;
    Framerate _framerate;
    luisa::unique_ptr<Sampler::Instance> _aux_sampler;
    luisa::vector<float4> _pixels;
    luisa::optional<Window> _window;

private:
    void _render_one_camera(
        CommandBuffer &command_buffer, Pipeline &pipeline,
        AuxiliaryBufferPathTracingInstance *pt, Camera::Instance *camera) noexcept;

public:
    explicit AuxiliaryBufferPathTracingInstance(const AuxiliaryBufferPathTracing *node, Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node},
          _aux_sampler{node->aux_sampler()->build(pipeline, cmd_buffer)} {
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<AuxiliaryBufferPathTracing>();
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

luisa::unique_ptr<Integrator::Instance> AuxiliaryBufferPathTracing::build(Pipeline &pipeline, CommandBuffer &cmd_buffer) const noexcept {
    return luisa::make_unique<AuxiliaryBufferPathTracingInstance>(this, pipeline, cmd_buffer);
}

void AuxiliaryBufferPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Pipeline &pipeline,
    AuxiliaryBufferPathTracingInstance *pt, Camera::Instance *camera) noexcept {

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
    command_buffer << synchronize();

    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;
    Callable balanced_heuristic = [](Float pdf_a, Float pdf_b) noexcept {
        auto p = pdf_a + pdf_b;
        return ite(p > 0.0f, pdf_a / p, 0.0f);
    };

    // 3 diffuse, 3 specular, 3 normal, 1 depth, 3 albedo, 1 roughness, 1 emissive, 1 metallic, 1 transmissive, 1 specular-bounce
    auto auxiliary_output = pipeline.device().create_image<float>(PixelStorage::FLOAT4, resolution);
    auto auxiliary_noisy = pipeline.device().create_image<float>(PixelStorage::FLOAT4, resolution);
    auto auxiliary_diffuse = pipeline.device().create_image<float>(PixelStorage::FLOAT4, resolution);
    auto auxiliary_specular = pipeline.device().create_image<float>(PixelStorage::FLOAT4, resolution);
    auto auxiliary_normal = pipeline.device().create_image<float>(PixelStorage::FLOAT4, resolution);
    auto auxiliary_depth = pipeline.device().create_image<float>(PixelStorage::FLOAT1, resolution);
    auto auxiliary_albedo = pipeline.device().create_image<float>(PixelStorage::FLOAT4, resolution);
    auto auxiliary_roughness = pipeline.device().create_image<float>(PixelStorage::FLOAT2, resolution);

    Kernel2D clear_kernel = [&]() noexcept {
        auxiliary_noisy.write(dispatch_id().xy(), make_float4(0.0f));
        auxiliary_diffuse.write(dispatch_id().xy(), make_float4(0.0f));
        auxiliary_specular.write(dispatch_id().xy(), make_float4(0.0f));
        auxiliary_normal.write(dispatch_id().xy(), make_float4(0.0f));
        auxiliary_depth.write(dispatch_id().xy(), make_float4(0.0f));
        auxiliary_albedo.write(dispatch_id().xy(), make_float4(0.0f));
        auxiliary_roughness.write(dispatch_id().xy(), make_float4(0.0f));
    };

    //Callable accumulate = [&](ImageFloat image, Expr<uint2> pixel, Expr<float3> rgb) noexcept {
    //    $if(!any(isnan(rgb) || isinf(rgb))) {
    //        auto pixel_id = pixel.y * resolution.x + pixel.x;
    //        auto threshold = node<ColorFilm>()->clamp();
    //        auto lum = srgb_to_cie_y(rgb);
    //        auto c = rgb * (threshold / max(lum, threshold));
    //        for (auto i = 0u; i < 3u; i++) {
    //            image.atomic(pixel_id * 4u + i).fetch_add(c[i]);
    //        }
    //        image.atomic(pixel_id * 4u + 3u).fetch_add(1.f);
    //    };
    //};

    pt->_aux_sampler->reset(command_buffer, resolution, pixel_count, node<AuxiliaryBufferPathTracing>()->noisy_count());
    command_buffer << synchronize();

    Kernel2D render_auxiliary_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);

        auto pixel_id = dispatch_id().xy();
        pt->_aux_sampler->start(pixel_id, frame_index);
        auto [camera_ray, camera_weight] = camera->generate_ray(*pt->_aux_sampler, pixel_id, time);
        auto spectrum = pipeline.spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : pt->_aux_sampler->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        $for(depth, pt->node<AuxiliaryBufferPathTracing>()->max_depth()) {

            // trace
            auto it = pipeline.geometry()->intersect(ray);

            $if(depth == 0 & it->valid()) {
                auxiliary_normal.write(dispatch_id().xy(), make_float4(it->shading().n(), 1.f));
                auxiliary_depth.write(dispatch_id().xy(), make_float4(length(it->p() - ray->origin()), 0.f, 0.f, 0.f));
                pipeline.surfaces().dispatch(it->shape()->surface_tag(), [&](auto surface) noexcept {
                    // create closure
                    auto closure = surface->closure(*it, swl, 1.f, time);
                    
                });
            };

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
            auto u_light_selection = pt->_aux_sampler->generate_1d();
            auto u_light_surface = pt->_aux_sampler->generate_2d();
            Light::Sample light_sample = light_sampler->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
            auto occluded = pipeline.geometry()->intersect_any(shadow_ray);

            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto u_lobe = pt->_aux_sampler->generate_1d();
            auto u_bsdf = pt->_aux_sampler->generate_2d();
            auto eta_scale = def(1.f);
            pipeline.surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(*it, swl, 1.f, time);
                if (auto dispersive = closure->dispersive()) {
                    $if(*dispersive) { swl.terminate_secondary(); };
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
                        auto w = balanced_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                    };

                    // sample material
                    auto sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(sample.wi);
                    pdf_bsdf = sample.eval.pdf;
                    auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);
                    beta *= w * sample.eval.f;

                    // apply eta scale
                    $switch(sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(sample.eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / sample.eta); };
                    };
                };
            });

            // rr
            $if(beta.all([](auto b) noexcept { return isnan(b) | b <= 0.f; })) { $break; };
            auto rr_depth = pt->node<AuxiliaryBufferPathTracing>()->rr_depth();
            auto rr_threshold = pt->node<AuxiliaryBufferPathTracing>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth & q < rr_threshold) {
                $if(pt->_aux_sampler->generate_1d() >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        auto curr = auxiliary_noisy.read(pixel_id);
        auxiliary_noisy.write(pixel_id, curr + make_float4(spectrum->srgb(swl, Li * shutter_weight), 1.f));
    };

    Kernel2D convert_image_kernel = [](ImageFloat accum, ImageFloat output) noexcept {
        auto pixel_id = dispatch_id().xy();
        auto curr = accum.read(pixel_id).xyz();
        auto scale = 1.f / accum.read(pixel_id).w;
        output.write(pixel_id, make_float4(scale * curr, 1.f));
    };

    auto convert_image = pipeline.device().compile(convert_image_kernel);

    Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel_id, time);
        auto spectrum = pipeline.spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        $for(depth, pt->node<AuxiliaryBufferPathTracing>()->max_depth()) {

            // trace
            auto it = pipeline.geometry()->intersect(ray);

            $if(depth == 0) {
                auxiliary_normal.write(dispatch_id().xy(), make_float4(it->shading().n(), 1.f));
            };

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
            auto u_light_selection = sampler->generate_1d();
            auto u_light_surface = sampler->generate_2d();
            Light::Sample light_sample = light_sampler->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto shadow_ray = it->spawn_ray(light_sample.wi, light_sample.distance);
            auto occluded = pipeline.geometry()->intersect_any(shadow_ray);

            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto u_lobe = sampler->generate_1d();
            auto u_bsdf = sampler->generate_2d();
            auto eta_scale = def(1.f);
            pipeline.surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(*it, swl, 1.f, time);
                if (auto dispersive = closure->dispersive()) {
                    $if(*dispersive) { swl.terminate_secondary(); };
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
                        auto w = balanced_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                    };

                    // sample material
                    auto sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(sample.wi);
                    pdf_bsdf = sample.eval.pdf;
                    auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);
                    beta *= w * sample.eval.f;

                    // apply eta scale
                    $switch(sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(sample.eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / sample.eta); };
                    };
                };
            });

            // rr
            $if(beta.all([](auto b) noexcept { return isnan(b) | b <= 0.f; })) { $break; };
            auto rr_depth = pt->node<AuxiliaryBufferPathTracing>()->rr_depth();
            auto rr_threshold = pt->node<AuxiliaryBufferPathTracing>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth & q < rr_threshold) {
                $if(sampler->generate_1d() >= q) { $break; };
                beta *= 1.0f / q;
            };
        };
        camera->film()->accumulate(pixel_id, spectrum->srgb(swl, Li * shutter_weight));
    };

    Clock clock_compile;
    auto clear_shader = pipeline.device().compile(clear_kernel);
    auto render_auxiliary = pipeline.device().compile(render_auxiliary_kernel);
    auto render = pipeline.device().compile(render_kernel);
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
    {
        std::ofstream file{"results.txt", std::ios::app};
        file << "Shader compile time = " << integrator_shader_compilation_time << " ms" << std::endl;
    }
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    LUISA_INFO("Rendering started.");
    Clock clock;
    ProgressBar progress;
    progress.update(0.0);

    auto dispatch_count = 0u;
    auto dispatches_per_commit = 32u;
    auto sample_id = 0u;
    auto auxiliary_sample_id = 0u;
    for (auto s : shutter_samples) {
        pipeline.update(command_buffer, s.point.time);
        command_buffer << clear_shader().dispatch(resolution);
        for (auto i = 0u; i < pt->node<AuxiliaryBufferPathTracing>()->noisy_count(); i++) {
            std::vector<float> hostaux_noisy(pixel_count * 4);
            std::vector<float> hostaux_normal(pixel_count * 4);
            std::vector<float> hostaux_depth(pixel_count * 1);
            command_buffer << render_auxiliary(auxiliary_sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution)
                           << convert_image(auxiliary_noisy, auxiliary_output).dispatch(resolution)
                           << auxiliary_output.copy_to(hostaux_noisy.data())
                           << auxiliary_normal.copy_to(hostaux_normal.data())
                           << auxiliary_depth.copy_to(hostaux_depth.data())
                           << synchronize();
            // noisy
            auto noisy_path = camera->node()->file().parent_path() / fmt::format("s{}_noisy.exr", i + 1);
            save_image(noisy_path, hostaux_noisy.data(), resolution);
            // normal
            auto nrm_path = camera->node()->file().parent_path() / fmt::format("s{}_normal.exr", i + 1);
            save_image(nrm_path, hostaux_normal.data(), resolution);
            // depth
            auto dep_path = camera->node()->file().parent_path() / fmt::format("s{}_depth.exr", i + 1);
            save_image(dep_path, hostaux_depth.data(), resolution, 1);
        }
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
    {
        std::ofstream file{"results.txt", std::ios::app};
        file << "Render time = " << render_time << " ms" << std::endl;
    }
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::AuxiliaryBufferPathTracing)
